import json
import time
from pathlib import Path
from typing import List, Callable, Optional
from datasets import load_dataset
from qdrant_client.models import PointStruct, VectorParams, Distance
from state.session_flow import SessionFlow
from state.watch_storage import WatchStorage
from vectorize.embeddings import Embed, AggregationModes
from vectorize.embeddings import EmbeddingVector, EmbeddingSourceType
from config import flags

import logging
log = logging.getLogger(__name__)


class HFIngest:
    """
    Ingests a Hugging Face dataset stored locally into a Qdrant vector database.
    
    The collection name in Qdrant is derived from the watch name.
    The dataset folder path is retrieved from the WatchStorage.
    The Qdrant client is obtained from the SessionFlow instance.
    """

    def __init__(self, session_flow: SessionFlow, watch_name: str):
        """
        Initialize the HFIngest connector.
        
        Args:
            session_flow: The SessionFlow instance that provides the Qdrant client
                          and other shared application state.
            watch_name: The name of the watch item in WatchStorage. The dataset
                       folder path is retrieved from the watch's path field.
        """
        self.session_flow = session_flow
        self.collection_name = watch_name
        
        # Retrieve the watch object and dataset folder path from WatchStorage
        watch_storage = WatchStorage(session_flow.watchlists_file_path)
        watch = watch_storage.get_watch(watch_name)
        if not watch:
            log.error(f"No watch found for '{watch_name}'")
            raise ValueError(f"No watch found for '{watch_name}'")
        
        dataset_folder = watch_storage.get_watch_folder(watch_name)
        if not dataset_folder:
            log.error(f"No dataset folder found for watch '{watch_name}'")
            raise ValueError(f"No dataset folder found for watch '{watch_name}'")
        
        # Get the embedding model name and vector name from the watch configuration
        self.vector_name = watch.embedding_vector if watch.embedding_vector else None
        self.vector_dimension = watch.vector_dimension if watch.vector_dimension else flags['embedding_dims']
        log.info(f"Vector name for collection: '{self.vector_name}', dimension: {self.vector_dimension}")
        
        self.dataset_folder = Path(dataset_folder)
        log.info(f"HFIngest initialized for dataset folder: {self.dataset_folder}")
        log.info(f"Qdrant collection name: '{self.collection_name}'")
        
        # Get the Qdrant client from SessionFlow
        self.qdrant_client = session_flow.qdrant_client
        log.info(f"Qdrant client obtained from SessionFlow")
        
        # Validate that the dataset folder exists
        if not self.dataset_folder.exists():
            log.error(f"Dataset folder does not exist: {self.dataset_folder}")
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_folder}")
        
        # Ensure metadata.jsonl files use 'file_name' key (required by HF ImageFolder)
        # Some datasets use 'image' instead — rename it before loading
        self._fix_metadata_file_name_key(self.dataset_folder)
        
        # Load the Hugging Face dataset from the local folder
        self.dataset = load_dataset(str(self.dataset_folder))
        log.info(f"Hugging Face dataset loaded from: {self.dataset_folder}")
        log.info(f"Dataset structure: {self.dataset}")

        # Initialize the embedding encoder via EmbeddingVector
        # Use the watch's embedding_vector as the model name if available, otherwise fall back to config
        model_name = self.vector_name if self.vector_name else flags['model_name']

        # Try to retrieve EmbeddingVector from the Embeddings registry; fall back to a default
        embeddings_registry = session_flow.get_embeddings()
        if model_name in embeddings_registry:
            embedding_vector = embeddings_registry.get(model_name)
        else:
            # Build a default EmbeddingVector for sentence_transformer models
            embedding_vector = EmbeddingVector(
                reference_name=model_name,
                source_type=EmbeddingSourceType.SENTENCE_TRANSFORMER,
                api_key_name="HUGGINGFACE_HUB_TOKEN",
                dimensions=[self.vector_dimension],
                is_local=False,
            )
        self.embed = Embed(embedding_vector, AggregationModes.MEAN_POOLING, target_dim=self.vector_dimension)
        log.info(f"Embedding model '{model_name}' loaded with dim={self.vector_dimension}")

    @staticmethod
    def _fix_metadata_file_name_key(dataset_folder: Path):
        """
        Ensure that metadata.jsonl files contain a 'file_name' key as required
        by the Hugging Face ImageFolder dataset loader. If entries use 'image'
        instead of 'file_name', rewrite the file with the corrected key.
        """
        for metadata_path in dataset_folder.rglob("metadata.jsonl"):
            lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                continue
            # Check the first line to see if it needs fixing
            first = json.loads(lines[0])
            if "file_name" in first:
                continue  # already correct
            if "image" not in first:
                continue  # nothing to fix
            
            log.info(f"Renaming 'image' -> 'file_name' in {metadata_path}")
            fixed_lines = []
            for line in lines:
                record = json.loads(line)
                if "image" in record and "file_name" not in record:
                    record["file_name"] = record.pop("image")
                fixed_lines.append(json.dumps(record, ensure_ascii=False))
            metadata_path.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")

    def _read_file_names_from_metadata(self) -> List[str]:
        """
        Read 'file_name' values directly from metadata.jsonl files in the dataset folder.
        This is needed because HF ImageFolder consumes the 'file_name' field and converts
        it into a PIL Image 'image' column, losing the original string value.
        
        Returns:
            List of file_name strings in the order they appear in metadata.jsonl.
        """
        file_names = []
        for metadata_path in sorted(self.dataset_folder.rglob("metadata.jsonl")):
            lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                record = json.loads(line)
                if "file_name" in record:
                    file_names.append(record["file_name"])
                else:
                    file_names.append("")
        log.info(f"Read {len(file_names)} file_name entries from metadata.jsonl")
        return file_names

    def _embed(self, text: str) -> List[float]:
        """
        Embed a text string into a vector using the configured embedding model.
        Uses chunking and mean pooling for texts longer than the model's token window.
        
        Args:
            text: The text string to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        return self.embed.encode(text)

    def prepare(self, num_batches: int, payload_fields: List[str], embedding_fields: List[str]):
        """
        Prepare the ingestion by creating/recreating the Qdrant collection and
        computing batch boundaries. Must be called before ingest_batch().
        
        Args:
            num_batches: The number of batches to split the dataset into.
            payload_fields: List of field names to store as payload in Qdrant.
            embedding_fields: List of field names to concatenate and embed as vectors.
        """
        # If the collection already exists, delete it first to avoid inconsistencies
        if self.qdrant_client.collection_exists(self.collection_name):
            log.info(f"Collection '{self.collection_name}' already exists — deleting before re-ingestion")
            self.qdrant_client.delete_collection(self.collection_name)
            log.info(f"Collection '{self.collection_name}' deleted successfully")
        
        # Create the collection with the appropriate vector configuration
        vector_params = VectorParams(size=self.vector_dimension, distance=Distance.COSINE)
        if self.vector_name:
            vectors_config = {self.vector_name: vector_params}
        else:
            vectors_config = vector_params
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )
        log.info(f"Collection '{self.collection_name}' created with vector dim={self.vector_dimension}")
        
        # Store batch configuration for use by ingest_batch()
        split_name = list(self.dataset.keys())[0]
        self._data = self.dataset[split_name]
        
        # HF ImageFolder converts 'file_name' into an 'image' column (PIL Image).
        # If 'file_name' is requested as a payload field but missing from the dataset,
        # reconstruct it by reading the original metadata.jsonl file directly.
        # We store it as a separate list since add_column() can fail on image datasets.
        self._file_names_override = None
        if 'file_name' in payload_fields and 'file_name' not in self._data.column_names:
            file_names = self._read_file_names_from_metadata()
            if file_names and len(file_names) == len(self._data):
                self._file_names_override = file_names
                log.info(f"Reconstructed 'file_name' from metadata.jsonl ({len(file_names)} entries)")
            else:
                log.warning(f"Could not reconstruct 'file_name': metadata has {len(file_names) if file_names else 0} "
                           f"entries but dataset has {len(self._data)} records")
        
        self._total_records = len(self._data)
        self._num_batches = num_batches
        self._batch_size = max(1, self._total_records // num_batches)
        self._payload_fields = payload_fields
        self._embedding_fields = embedding_fields
        self._first_batch_time = None
        
        log.info(f"Prepared ingestion: {self._total_records} records in {num_batches} batches (batch_size={self._batch_size})")

    def ingest_batch(self, batch_idx: int) -> tuple:
        """
        Ingest a single batch into Qdrant and return progress information.
        
        Args:
            batch_idx: The zero-based index of the batch to process.
            
        Returns:
            tuple: (progress: float, time_remaining: str) where progress is 0.0 to 1.0
        """
        batch_start = batch_idx * self._batch_size
        batch_end = min(batch_start + self._batch_size, self._total_records)
        
        # Last batch picks up remaining records
        if batch_idx == self._num_batches - 1:
            batch_end = self._total_records
        
        if batch_start >= self._total_records:
            return (1.0, "0s")
        
        batch_start_time = time.time()
        
        batch_data = self._data[batch_start:batch_end]
        points = []
        
        # Process each datapoint in the batch
        num_points = batch_end - batch_start
        for i in range(num_points):
            # Build payload from payload_fields
            payload = {}
            for field in self._payload_fields:
                # Use the file_names_override for 'file_name' if HF consumed it
                if field == 'file_name' and self._file_names_override is not None:
                    payload[field] = self._file_names_override[batch_start + i]
                elif field in batch_data:
                    value = batch_data[field]
                    payload[field] = value[i] if isinstance(value, list) else value
            
            # Build text for embedding from embedding_fields
            text_parts = []
            for field in self._embedding_fields:
                if field in batch_data:
                    value = batch_data[field]
                    field_value = value[i] if isinstance(value, list) else value
                    if field_value is not None:
                        text_parts.append(str(field_value))
            
            embedding_text = " ".join(text_parts)
            vector = self._embed(embedding_text)
            
            # Use named vector if the collection was created with named vectors
            if self.vector_name:
                vector_data = {self.vector_name: vector}
            else:
                vector_data = vector
            
            points.append(PointStruct(
                id=batch_start + i,
                vector=vector_data,
                payload=payload,
            ))
        
        # Write batch to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        batch_elapsed = time.time() - batch_start_time
        
        # After first batch, compute time estimates
        if batch_idx == 0:
            self._first_batch_time = batch_elapsed
        
        # Update progress
        progress = (batch_idx + 1) / self._num_batches
        
        # Compute remaining time
        if self._first_batch_time is not None:
            remaining_batches = self._num_batches - (batch_idx + 1)
            remaining_seconds = remaining_batches * self._first_batch_time
            
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            
            if hours > 0:
                time_remaining = f"{hours}h {minutes:02d}m {seconds:02d}s"
            elif minutes > 0:
                time_remaining = f"{minutes}m {seconds:02d}s"
            else:
                time_remaining = f"{seconds}s"
        else:
            time_remaining = "Estimating..."
        
        log.info(f"Batch {batch_idx + 1}/{self._num_batches} completed - "
                 f"Progress: {progress:.1%} - Time remaining: {time_remaining}")
        
        return (progress, time_remaining)

    @property
    def num_batches(self) -> int:
        """Return the number of batches configured by prepare()."""
        return self._num_batches

    @property
    def total_records(self) -> int:
        """Return the total number of records in the dataset."""
        return self._total_records

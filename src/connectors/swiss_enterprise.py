from datasets import Features, load_dataset
import json
import re
import time
import os
import sys
import traceback
from vectorize.vector_db import VectorDBInterface
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from config import flags
from vectorize.embeddings import Embed, AggregationModes, Embeddings
from connectors.amtsblattportal import AmtsblattXML
from connectors.state import State
from connectors.hf_bootstrap import Bootstrap


import logging
log = logging.getLogger()

class SwissEnterpriseDB:
    # def __init__(
    #     self,
    #     collection_name: str = "swiss_enterprises",
    #     host: str = "localhost",
    #     port: int = 6333,
    #     url: Optional[str] = None,
    #     api_key: Optional[str] = None,
    #     features: Features = None
    # ):
        

    def __init__(self, db: VectorDBInterface, embedding_model_name: str, features: Features ):
        """
        Initialize SwissEnterprises with a vector database implementation.
        
        Args:
            db: An instance of a class implementing VectorDBInterface
                   (e.g., Qdrant)
            embedding_model_name: The HF string of the model for SentenceTransformer
            features: The fields that define the db (metadata_schema) without the embedding vector

        Raises:
            TypeError: If qdrant doesn't implement VectorDBInterface
        """
        super().__init__()
        # Verify that the provided object implements VectorDBInterface
        if not isinstance(db, VectorDBInterface):
            raise TypeError(
                f"db must implement VectorDBInterface, "
                f"got {type(db).__name__}"
            )
        
        self.db = db
        embeddings_registry = Embeddings()
        embedding_vector = embeddings_registry.get(embedding_model_name)
        target_dim = flags['embedding_dims']
        self.embed = Embed(embedding_vector, AggregationModes.MEAN_POOLING, target_dim)
        self.collection_name = "swiss_enterprises" 
        self.features = features       
        self.payload = self._hf_features_to_qdrant_payload(features)

        if not self.db.collection_exists(self.collection_name):
            self.db.create_collection(self.collection_name)
            log.info(f"{self.collection_name} collection is created.")

    # def embeddings(self, dense_model):
    #     # Load dense embedding model
    #     log.info(f"use {dense_model}")
    #     self.dense_encoder = SentenceTransformer(dense_model)
    #     log.info(f" {dense_model} is loaded")
    #     self.dense_dim = self.dense_encoder.get_sentence_embedding_dimension()


    def _uid_to_number(self,uid: str) -> int:
        """
        Convert a Swiss UID (e.g., 'CHE-123.456.789') into a int.

        Args: 
            the UID

        Returns:
            int value of UID 

        """
        # Extract digits only
        digits = re.sub(r"\D", "", uid)
        return int(digits)


    def _number_to_uid(self,number: int) -> str:
        """
        Convert an int into Swiss UID format ('CHE-123.456.789').

        Args:
            int value of UID

        Returns:
            The string value of UID    

        """
        number = re.sub(r"\D", "", str(number))  # keep only digits
        if len(number) != 9:
            raise ValueError("UID numeric part must be exactly 9 digits.")
        return f"CHE-{number[0:3]}.{number[3:6]}.{number[6:9]}"


    def create_collection(self, name, dimension, metadata_schema) -> bool:
        """
        Create a collection with dense vectors. 
        
        Args: 
            name: The name of the DB collection
            dimension: The embedding vector dimension, could define Matryosha truncated dimension
            metadata_schema: This defines the fields of the record (embedding vector is not included)
        
        """
        return self.db.create_collection(name, dimension, metadata_schema)
        # self.client.recreate_collection(
        #     collection_name=self.collection_name,
        #     vectors_config=rest.VectorParams(
        #         size= self.dense_dim,
        #         distance=rest.Distance.COSINE,
        #     ),            
        #)



    # def _encode_dense(self, text: str) -> List[float]:
    #     """Generate dense embedding from text."""
    #     return self.dense_encoder.encode(text, show_progress_bar=False).tolist()


    def load_bootstarp(self, hf_file_path, limit=None) -> bool:
        """
        Loads the bootstap dataset into Qdrant collection. 
        This bootstrap dataset was scraped before the existance of Amtsblattportal.
        Automatically generates dense vectors from text fields.

        Args: 
            hf_file_path: Path to the HuggingFace dataset file

        Returns:
            True if done    


        """
        log.debug(f"{self.features}")

        # load the Hugging Face dataset format
        dataset = load_dataset(
            "csv",
            data_files=hf_file_path,
            delimiter="\t",
            encoding="utf-8",
            column_names=list(self.features.keys())
        )

        # if limit is define
        if limit is not None:
            log.info(f"The select {limit} row of bootstrap are uploaded and ingested.")
            dataset = dataset['train'].select(range(limit))
        else: 
            log.info(f"The whole bootstrap is uploaded and ingested.")
            dataset = dataset['train']

   
        slice_size= flags['boot_slice_size']
        total_size = dataset.num_rows
        log.info(f"Dataset will be loaded in slices of {slice_size} each from a total of {total_size}")      
        start_time = time.perf_counter()

        for slice_num, start_idx in enumerate(range(0, total_size, slice_size)):
            end_idx = min(start_idx + slice_size, total_size)
            slice_data = dataset.select(range(start_idx,end_idx))         

            vectors = []
            # count =0
            # nb_to_log = int(slice_size * 0.001)
            # if nb_to_log == 0 : nb_to_log = 100
            for row in slice_data:
                text = f"{row['name']} {row['purpose']} {row['web_summary']}"
                vector = self.embed.encode(text)
                vectors.append(vector)
                # count +=1
                # if count%nb_to_log == 0: 
                #     log.info(f"{count} are vectorized")

            # vectors = self.embed.encode(
            #     [f"{row['name']} {row['purpose']} {row['web_summary']}" for row in data],truncate_dim=flags['embedding_dims']
            # )

            payloads = [dict(row) for row in slice_data]

            records = [
                {**payload, "vector": vector}
                for vector, payload in zip(vectors, payloads)            
            ]

            uids = [self._uid_to_number(row["uid"]) for row in slice_data]

            if not self.db.upload_records(self.collection_name,records=records,record_ids=uids):
                log.error(f"The records were not uploaded.")
                return False
            
            log.info(f"Completed slice {slice_num + 1}: indices {start_idx}-{end_idx}")
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            log.info(f"Execution time so far is of {int(execution_time/60)} minutes")

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        log.info(f"Total execution time : {int(execution_time/60)} minutes")
        return True    

        # self.client.upload_collection(
        #     collection_name=self.collection_name,
        #     vectors=embeddings,
        #     payload=payloads,
        #     ids=uids, 
        #     batch_size=256,
        #     # show_progress=False,
        # )

        # log.info(f" Done! Points count: {self.client.get_collection('swiss_enterprises').points_count}")
        # log.info(f" Kiodia's record: {self.client.retrieve(collection_name='swiss_enterprises', ids=[self._uid_to_number('CHE-405.656.730')])}")


        # Generate vectors from text
        # text_for_embedding = f"{name}. {purpose}. {summary}"
        # dense_vector = self._encode_dense(text_for_embedding)

        # self.client.upsert(
        #     collection_name=self.collection_name,
        #     points=[
        #         rest.PointStruct(
        #             payload=self._hf_features_to_qdrant_payload(self,self.features),
        #             vector={
        #                 "dense": dense_vector,
        #             },
        #         )
        #     ],
        # )


    def uid_exists_by_id(self, uid: str) -> bool:
        uid_int = int(re.sub(r"\D", "", uid))  # "CHE-405.656.730" → 405656730
        return self.db.record_exists(collection_name=self.collection_name, record_id=uid_int)


    

    def grep_simple(self,filename, pattern) -> bool:
        ''' 
        Log.info the grep result throug the *.tsv dataset

        Arg: 
            filename: The file path of the dataset
            pattern: The grep pattern to search

        Returns:
            True if found

        '''
        found = False
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if pattern in line:
                    log.info(f"The {pattern} exists in {filename} at {line_num}: {line}")
                    found = True                

        if not found:
            log.info(f"The {pattern} is missing in {filename}")

        return found    

    def get_record(self,UID:str) -> Dict[str,Any]:
        '''
        Return a record given its UID

        Arg:
            UID : the string CHE-xxx.xxx.xxx

        Returns: 
            the Dict key, value of the record 
            'id': the integer index corresponding the UID 
            'vector': the embedding vector List[float]
            'metadata': the content of the record Dict[str,str]
        
        '''
        id = self._uid_to_number(UID)
        return self.db.get_record(self.collection_name,id)


    def query(self,query_text:str) -> List[Dict[str,str]]:

        log.info(f"The query is encoded: {query_text}")
        query_vector = self.embed.encode(query_text)

        match_contitions = {'status':'active'}

        results = self.db.search_similar(
            self.collection_name,
            query_vec=query_vector,
            top_k=2, 
            filter_conditions=match_contitions)
              
        # results = self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector=query_vector,
        #     query_filter=rest.Filter(
        #         must=[
        #             rest.FieldCondition(
        #                 key="status",
        #                 match=rest.MatchValue(value="active")
        #             )
        #         ]
        #     ),
        #     limit=1,
        #     with_payload=True
        # )

        log.info("results are returned")
        return results
    

    def amtsblatt_xmlfile_updates(self,start_day:datetime, end_day:datetime, state):
        '''
        From the available Amtsblatt portal downloaded XML files, ingest updates into the vector DB

        Arg: 
            start_date, the date from which we start to ingest, if needed according to state.latest_day
            end_data, the date from which we stop to ingest
            state indicates the latest_data of data already ingested

        '''
        dest_dir = flags['dest_dir']
        log.info(dest_dir)
        am = AmtsblattXML(dest_dir)


        try: 
            log.info(f"Start day {start_day.strftime("%d-%m-%Y")}")
            log.info(f"End day {end_day.strftime("%d-%m-%Y")}")        

            # Loop through each day from start_day to end_day (inclusive)
            current_day = start_day
            while current_day <= end_day:
                # Your code here for each day
                log.info(f"Processing day: {current_day.strftime('%Y-%m-%d')}")

                # Daily update
                am.update_day(current_day,self)  # update from Amtsblatt portal            
                
                # Move to next day
                current_day += timedelta(days=1)
                    
        except Exception as e:
            message = f"Error to ingest Amtsblatt's XML files {str(e)}"
            log.error(message)


    def ingest_entries_before_AmtsblattXML (self, head_limit:int ):
        '''
        Ingest the Swiss Enterprise entries before the existance of AmtsblattXML portal.

        To regenerate the bootstap dataset set the state of bootstrap to False
        and remove the hf_fosc_boot file

        Args:
            head_limit: Enable a small run typically 1500 of entries among the 600K ones
        
        
        '''
        state = State(flags['dest_dir'])
        # the date from 2018-09-03 till 2025 where already uploaded
        # if we start in 2025 we save time starting in 2025
        #state.start_day = datetime(year=2025, month=10, day=2)
        log.info(f"state start day {state.start_day}")

        if not state.bootstrapped_state :
            # ingest datasets
            if not os.path.exists(flags['hf_fosc_boot']):
                boot = Bootstrap(flags['source_fosc_boot'],flags['hf_fosc_boot'])
                added_length = len(flags['fosc_hf_field_description'])-len(flags['fosc_field_description'])
                # the bootstrap have been created on the 2018-09-03
                # the records get that day as creation date
                boot_date = datetime(year=2018, month=9, day=3) 
                creation_date = boot_date.strftime("%Y-%m-%d")
                boot.get_hf_fosc(added_length, creation_date)
                log.info(f"The added field to payload: {added_length}")


            self.load_bootstarp(flags['hf_fosc_boot'], head_limit)
            state.bootstrapped_state = True        



    def add_new_entry(self, soup) -> int:
        ''' 
        Add a new data point with usigned integer uid as id index
        
        Arg: 
            soup the beatiful soup xml of the Amstblatt portal xml HR01 publication

        Returns: 
            True is te pattern is found
        
        '''        
        name = soup.find('company').find('name').text
        log.debug(f"New company {name}")

        def get_text(tag_name, parent=soup):
            tag = parent.find(tag_name)
            return tag.text.strip() if tag and tag.text else None

        company = soup.find("company")
  
        # --- Build the record ---
        uid_int = int(get_text("uidOrganisationId", company))
        uid = self._number_to_uid(uid_int)
        publication_date= get_text("publicationDate", soup)
        record = {
            "uid": uid,
            "name": get_text("name", company),
            "status": "active",
            "legal_seat": get_text("seat", company),
            "legal_form": get_text("legalForm", company),
            "register_office": get_text("officeName", soup.find("senderOffice")),
            "publication_dates": [publication_date], # always the first one in the list
            "noga_code": None,
            "address_street": get_text("street", company),
            "address_city": get_text("town", company),
            "link": f"https://www.zefix.admin.ch/fr/search/entity/list?name={uid.replace('-', '').replace('.', '')}&directLink=true",
            "purpose_language": get_text("language", soup),
            "purpose": get_text("purpose", soup),
            "web_link": None,
            "web_summary": None,
            "web_timestamp": None,
            # Store current Swiss timestamp (CET/CEST aware)
            # "creation_timestamp": datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d"),
            "creation_timestamp": publication_date,
        }

        # --- Create the embedding vector ---
        text_to_embed = f"{record['name']}  {record['purpose']}  {record['web_summary']}"
      
        # Optional: delete existing record first (safe no-op if it doesn’t exist)
        # not needed with Qdrant
        # self.client.delete(collection_name=self.collection_name, points_selector={"points": [uid_int]})

        # --- Upsert record ---
        record['vector']= self.embed.encode(text_to_embed)
        response = self.db.upsert_record(collection_name=self.collection_name, record_id=uid_int, updates=record)

        log.debug(f"Upsert completed for ID: {uid_int}")
        log.debug(f"Response: {response}")

        return uid_int


   
    def delete_entry(self, soup):
        ''' 
        Delete an exist or not data point with usigned integer uid as id
        
        Arg: 
            soup the beatiful soup xml of the HR03 Amstblatt portal xml publication
        
        '''
        name = soup.find('company').find('name').text
        log.debug(f"Deletion of the company name {name}")

        uid_tag = soup.find("uidOrganisationId")
        if uid_tag is None:
            raise ValueError("No <uidOrganisationId> tag found in the XML")

        # Extract as signed integer
        uid_int = int(uid_tag.text.strip())
        log.debug(f"Extracted UID (int): {uid_int}")

        response= self.db.delete_record(collection_name=self.collection_name,record_id=uid_int)

        log.debug(f"Delete response: {response}")



    def mutate_entry(self, soup):
        ''' 
        Mutate an exist or not data point with usigned integer uid as id
        
        Arg: 
            soup the beatiful soup xml of the HR02 Amstblatt portal xml publication

        '''

        name = soup.find('commonsNew').find('company').find('name').text
        log.debug(f"Mutation of the company {name}")

        def get_text(tag, parent=soup):
            t = parent.find(tag)
            return t.text.strip() if t and t.text else None

        def get_publications_dates(uid_int, data_point, soup):
            """
            Checks if a point ID exists and extracts the specific field .
            """
            publication_dates = [get_text("publicationDate", soup)]                    

            # Test if the ID exists (the result list will contain one point)
            if not data_point:
                # => no previous data point we return the current publications 
                return publication_dates
            
            # we have one => get the previous publication dates
            payload = data_point['metadata']
            field_name = "publication_dates"

            # Extract the previous publications dates
            if payload and field_name in payload:
                if isinstance(payload.get(field_name), list) and len(payload[field_name]) > 0:
                    # the field contains previous publication dates =>
                    extended_dates = payload[field_name]
                    # not needed anymore if publication_dates[0] not in extended_dates: 
                    extended_dates.append(publication_dates[0])
                    log.debug(f"{uid_int} The extended publication {type(payload[field_name])} dates with previous dates: {extended_dates}")
                    # remove and duplication
                    return list(dict.fromkeys(extended_dates))
                    
                else:
                    # the field doesn't contain any previous publication dates => return the curreent one 
                    log.info(f"No privious publications")
                    return publication_dates
            else:
                log.error(f"The playload of {uid_int} does not have any pulbication_dates")
                return publication_dates        
        
        def get_previous_publication(uid_int) -> Dict[str,Any]:
            try:
                # retrieve any existing uid with its payload
                return self.db.get_record(collection_name=self.collection_name,record_id=uid_int)

            except Exception as e:
                log.error(f"Error in getting the previous publication {uid_int} an error occurred: {e}")
                # traceback.print_exc()
                # sys.exit(1)    

        def get_creation_date(uid,data_point):
            if not data_point:
                # => no previous data point 
                log.info(f"for {uid} no previous record was found.")
                self.grep_simple(flags['hf_fosc_boot'], uid)
                return "2018-09-03" # by default the date of the bootstrap
            
            # keep the creation date of the previous existing UID
            return data_point['metadata']["creation_timestamp"]                

        commons_new = soup.find("commonsNew")
        company = commons_new.find("company")


        # --- Extract fields from <commonsNew> ---
        uid_int = int(get_text("uidOrganisationId", company))
        data_point = get_previous_publication(uid_int)        
        uid = self._number_to_uid(uid_int)
        record = {
            "uid": uid,
            "name": get_text("name", company),
            "status": "active",  # HR02 = mutation
            "legal_seat": get_text("seat", company),
            "legal_form": get_text("legalForm", company),
            "register_office": get_text("officeName", soup.find("senderOffice")),
            "publication_dates": get_publications_dates(uid_int,data_point,soup),
            "noga_code": None,
            "address_street": get_text("street", company),
            "address_city": get_text("town", company),
            "link": f"https://www.zefix.admin.ch/fr/search/entity/list?name={uid.replace('-', '').replace('.', '')}&directLink=true" ,
            "purpose_language": get_text("language", soup),
            "purpose": get_text("purpose", commons_new),
            "web_link": None,
            "web_summary": None,
            "web_timestamp": None,
            # Swiss timestamp (timezone aware)
            #"creation_timestamp": datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d"),
            "creation_timestamp": get_creation_date(uid,data_point),            
        }

        # --- Compute embedding vector if necessary ---
        new_text = f"{record['name']}  {record['purpose']}  {record['web_summary']}"

        # Compute embedding only if needed
        if not data_point or f"{data_point['metadata']['name']}  {data_point['metadata']['purpose']}  {data_point['metadata']['web_summary']}" != new_text:
            record['vector'] = self.embed.encode(new_text)
        else:
            record['vector'] = data_point['vector']

        response = self.db.upsert_record(collection_name=self.collection_name,record_id=uid_int,updates=record)
  
        # --- Delete previous record if exists (no error if not found) ---
        # this is not necessary, it's the default behavior of upsert method
        # self.client.delete(collection_name=self.collection_name, points_selector={"points": [uid_int]})

        # --- Upsert into Qdrant ---
        # response = self.client.upsert(
        #     collection_name=self.collection_name,
        #     points=[
        #         {
        #             "id": uid_int,  # Unsigned integer UID
        #             "vector": embedding,
        #             "payload": record
        #         }
        #     ],
        # )

        log.debug("HR02 Mutation stored in Qdrant")
        log.debug(f"Data point ID: {uid_int}")
        log.debug("Response: {response}")  

        return response
    

    ''' 

    --- Private methods --- 
    
    '''


    def _hf_features_to_qdrant_payload(self,features) -> dict:
        # Convert Hugging Face Example to a Qdrant-safe payload
        payload = {}
        for key, value in features.items():
            # Ensure JSON-serializable
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                # Handle non-serializable types (e.g., lists of NumPy types)
                if hasattr(value, "tolist"):
                    payload[key] = value.tolist()
                else:
                    payload[key] = str(value)
        return payload
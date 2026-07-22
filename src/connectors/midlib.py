from qdrant_client import QdrantClient
from qdrant_client.http.models import ScrollRequest
from datasets import load_dataset
import os
import json
import requests
import shutil

import logging
log = logging.getLogger(__name__)


class Midlib:
    '''
    Error in the data set on image is duplicated:
     from the 4028 the following image is duplicated 662a74af22543caa20b216cc_Edmund_Leighton_V6_p.jpeg
     The embedding used are : https://github.com/qdrant/demo-midlibrary-explorer-nextjs 
     in the file encoder.py we have  cls.model = SentenceTransformer('clip-ViT-L-14')
     https://huggingface.co/openai/clip-vit-large-patch14
    
     The base model uses a ViT-L/14 Transformer architecture as an image encoder 
     and uses a masked self-attention Transformer as a text encoder. 
     These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

    The original implementation had two variants: one using a ResNet image encoder and the other using a Vision Transformer. 
    This repository has the variant with the Vision Transformer.

    '''

    name = 'midjourney'

    root_dataset = r'C:/Users/vankomme/datasets/watch_lists/midjourney' 
    meta_file_name = os.path.join(root_dataset,"metadata.jsonl")  

    def __init__(self):
        print("Init mid lib")

    def hf_download(self):
        print("Extract from midlib dataset by reading the one stored midjourney collection of Qdrant")

        print(self.meta_file_name)
        if os.path.exists(self.meta_file_name):
            os.remove(self.meta_file_name)    

        # Connect to Qdrant (use host/port for remote or just path for local file-based Qdrant)
        client = QdrantClient(host="localhost", port=6333)

        collection_name = "midjourney"
        scroll_offset = None  # Starting point

        
        # Get collection info
        info = client.get_collection(collection_name=collection_name)

        # Get number of points
        num_points = info.points_count
        print(f"The number of data points {num_points}")

        i=0

        my_set = set()

        with open(self.meta_file_name, "a", encoding="utf-8") as f:

            # for each point in Qdrant midjourney collection
            while True:
                
                response = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None,  # You can add a filter if needed
                    offset=scroll_offset,
                    limit=1,  # Read one point at a time
                    with_vectors=True
                )

                points = response[0]
                scroll_offset = response[1]

                # this doesn't work
                # if not points:
                #     break

                point = points[0]
                #print(f"ID: {point.id}, Vector: {point.vector}, Payload: {point.payload}")
                # print(f"id: {point.id}")

                v = point.vector
                p = point.payload

                d={}                
                d['id'] = point.id
                d['vector'] = v
                d['file_name'] = p['file_name'] 
                d['name']=p['name']
                d['url']=f"https://midlibrary.io{p['url']}"

                if not p['file_name'] in my_set:
                    my_set.add(p['file_name'])
                else :
                    print(f" from the {point.id} the following image is duplicated {p['file_name']}")                   

                json.dump(d, f)
                f.write("\n") 

                # Download and save the image if not done yet
                image_url = p['image_url']
                local_path = os.path.join(self.root_dataset, p['file_name'])

                if not os.path.exists(self.meta_file_name):
                    # the image is not loaded yet
                    print(image_url)
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        with open(local_path, 'wb') as f_image:
                            f_image.write(response.content)

                        # print(f"Image saved to {local_path}")
                    else:
                        print(f"Failed to download image. Status code: {response.status_code}")  

                i +=1
                if i >= num_points : 
                    print(f" break at the value of index {i}")
                    break


    def zipped(self):
        '''
        
        To speed up internal processin
        dataset.save_to_disk("my_dataset")
        import shutil
        shutil.make_archive("my_dataset", 'zip', "my_dataset")
        pros: 
        - Keeps fast Arrow format for fast reloading
        - Compresses well with .zip or .tar.gz
        - Great for backup, sharing, or later use with load_from_disk()

        '''
        shutil.make_archive(self.root_dataset, 'zip', self.root_dataset)


    def hf_info(self):
        print(f"Get hugging face from the dataset")
        dataset_midlib = load_dataset(self.root_dataset)
        print(dataset_midlib.shape)
        print(f"first item {dataset_midlib['train'][0]}")   



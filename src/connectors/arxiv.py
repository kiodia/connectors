import json
import os
from datetime import datetime

import logging
log = logging.getLogger(__name__)


class Arxiv :
    name = 'arxiv'
    dataset_path=r'C:\Users\vankomme\datasets\arxiv-json-archive\arxiv-metadata-oai-snapshot.json'
    hf_dataset = r'C:/Users/vankomme/datasets/watch_lists/arxiv.json'

    
    def __init__(self):
        print(f"Init dataset {self.name} at path {self.dataset_path}")

    # returns a list of dictonaries 
    def head (self, number):
        l = []
        if number <= 0 : return l

        with open(self.dataset_path) as file:
            for i, data_line in file:
                d= json.loads(data_line.rstrip())
                l.append(d)

                if number == i+1 : break

        return l
    
    def hf_extract(self, number, date):
        '''
        How to create a HF dataset => https://huggingface.co/docs/datasets/en/create_dataset

        '''
        if os.path.exists(self.hf_dataset):
            os.remove(self.hf_dataset)  

        with open(self.hf_dataset, "a", encoding="utf-8") as hf_file: 
            with open(self.dataset_path) as file:
                i=0
                for data_line in file:
                    d= json.loads(data_line.rstrip())
                    versions = d['versions']
                    v1 = versions[0]
                    publication_date=v1['created']
                    # datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                    create_date= datetime.strptime(publication_date, "%a, %d %b %Y %H:%M:%S %Z").date()
                
                    if create_date > date :
                        json.dump(d, hf_file)
                        hf_file.write("\n") 
                        # print(f"Number of publication {i}")
                        i+=1
                        if i >= number : break
        
        print(f"Path {self.hf_dataset}")
        # Lazy import: loading HF datasets before torch breaks torch's DLL
        # initialization on Windows, so it must not happen at module import.
        from datasets import load_dataset

        # "json", data_files="my_file.json")
        dataset = load_dataset("json", data_files= self.hf_dataset)
        print(dataset.shape)
        print(f"first item {dataset['train'][0]}")     
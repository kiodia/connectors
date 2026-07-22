
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import nltk
from nltk.corpus import reuters
import os
import pickle

'''
Hugging face load_dataset is not used for this reason:
ValueError: The repository for reuters21578 contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/reuters21578.
Please pass the argument `trust_remote_code=True` to allow custom code to be run.

ntlk is used instead

'''
class Reuters :
    name = 'reuters'

    dataset_path_corpus=r'C:\Users\vankomme\datasets\reuters\reuters.hf'
    dataset_path = r'C:\Users\vankomme\datasets\reuters'

            
    def __init__(self):

        # Load the dataset from file or from hugging face
        if not os.path.exists(self.dataset_path_corpus):

            # get the Reuters dataset ressource from NLTK
            nltk.download('reuters') 
            reuters_data = self.prepare_reuters_data()   
            # Step 3: Split the data into train and test sets
            train_data = {"id":[],"text": [], "labels": []}
            test_data = {"id":[],"text": [], "labels": []}

            for id, text, labels, split in zip(reuters_data["id"],reuters_data["text"], reuters_data["labels"], reuters_data["split"]):
                if split == "train":
                    train_data["id"].append(id)
                    train_data["text"].append(text)
                    train_data["labels"].append(labels)
                elif split == "test":
                    test_data["id"].append(id)                      
                    test_data["text"].append(text)
                    test_data["labels"].append(labels)

            # Step 4: Convert to Hugging Face Dataset
            train_dataset = Dataset.from_dict(train_data)
            test_dataset = Dataset.from_dict(test_data)                    

            # Combine into a DatasetDict for compatibility
            self.reuters_hf_dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })

            # print('Train and Test content')
            # print(self.reuters_hf_dataset["train"][0])
            # print(self.reuters_hf_dataset["test"][0])            

            # Step 5: Save the dataset locally in Hugging Face format
            self.reuters_hf_dataset.save_to_disk(self.dataset_path_corpus)                                         
                   
            print("The dataset doesn't exist locally and is now copied!")
            # Writing to a file
            # with open(self.dataset_path_corpus, "wb") as file:  # 'wb' is for write-binary mode
            #     pickle.dump(self.dataset_corpus, file)                   

        else:
            print('the dataset is available locally.')
            self.reuters_hf_dataset= load_from_disk(self.dataset_path_corpus)
            # with open(self.dataset_path_corpus, "rb") as file:  # 'rb' is for read-binary mode
            #     self.dataset_corpus = pickle.load(file)

        # print(str(type(self.reuters_hf_dataset)))
        # print('as read from disk:')
        # print(self.reuters_hf_dataset["train"][0])
        # print(self.reuters_hf_dataset["test"][0]) 
        # self.test_set = load_dataset(self.dataset_path_corpus, split="test") 
        #print(str(len(self.test_set[0])))        


    # Step 2: Organize the dataset into a dictionary
    def prepare_reuters_data(self):
        data = {"id":[],"text": [], "labels": [], "split": []}
        
        for file_id in reuters.fileids():
            # Get the id of the file
            id = file_id.split("/")[1]
            # Get the raw text
            text = reuters.raw(file_id)
            # Get the categories (labels)
            labels = reuters.categories(file_id)
            # Determine the split (training or test)
            split = "train" if file_id.startswith("training/") else "test"
            
            # Append to data
            data["id"].append(id)
            data["text"].append(text)
            data["labels"].append(labels)
            data["split"].append(split)
        
        return data


    def get_dataset_corpus(self):
        return self.reuters_hf_dataset
    

    def get_path(self):
        return self.dataset_path

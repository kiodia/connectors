from datasets import load_dataset
import os
import pickle

class Scifact :
    name = 'scifact'

    dataset_path_corpus=r'C:\Users\vankomme\datasets\scifact\scifact_corpus.pkl'
    dataset_path_queries=r'C:\Users\vankomme\datasets\scifact\scifact_queries.pkl'
    dataset_path_train=r'C:\Users\vankomme\datasets\scifact\scifact_train.pkl'        
    
    def __init__(self):

        # Load the BEIR/scifact dataset from file or from hugging face
        if not os.path.exists(self.dataset_path_corpus):
            self.dataset_corpus = load_dataset("BeIR/scifact","corpus", split="corpus")
            self.dataset_queries = load_dataset("BeIR/scifact", "queries", split="queries")
            print(str(len(self.dataset_queries)))  
            self.query_qrels = load_dataset("BeIR/scifact-qrels", split="train")  
            print(str(self.query_qrels[0]) )                    
            print("File doesn't exist!")
            # Writing to a file
            with open(self.dataset_path_corpus, "wb") as file:  # 'wb' is for write-binary mode
                pickle.dump(self.dataset_corpus, file)   
            with open(self.dataset_path_queries, "wb") as file:  # 'wb' is for write-binary mode
                pickle.dump(self.dataset_queries, file)                        
            with open(self.dataset_path_train, "wb") as file:  # 'wb' is for write-binary mode
                pickle.dump(self.query_qrels, file)                

        else:
            print('the dataset exists.')
            with open(self.dataset_path_corpus, "rb") as file:  # 'rb' is for read-binary mode
                self.dataset_corpus = pickle.load(file)
            with open(self.dataset_path_queries, "rb") as file:  # 'rb' is for read-binary mode
                self.dataset_queries = pickle.load(file)
            with open(self.dataset_path_train, "rb") as file:  # 'rb' is for read-binary mode
                self.query_qrels = pickle.load(file)                                


    def get_dataset_corpus(self):
        return self.dataset_corpus

    def get_dataset_queries(self):
        return self.dataset_queries

    def get_dataset_train(self):
        return self.query_qrels 

    def head(self, number):
        return self.dataset[number]      
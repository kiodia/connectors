import os
import json
import csv
import requests
import pickle
from datasets import load_dataset, Dataset

import logging
log = logging.getLogger(__name__)

# Here is your key: 7282a1f6
# Please append it to all of your API requests,
# OMDb API: http://www.omdbapi.com/?i=tt3896198&apikey=7282a1f6
#
# Generating train split: 0 examples [00:00, ? examples/s]
# Generating train split: 971 examples [00:00, 15835.32 examples/s]
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'Title', 'Year', 'Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer', 'Actors', 'Plot', 'Language', 'Country', 'Awards', 'Poster', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes', 'imdbID', 'Type', 'DVD', 'BoxOffice', 'Production', 'Website', 'Response'],
#         num_rows: 971
#     })
# })

class IMDB :

    root_dataset = r"C:\Users\vankomme\datasets\IMDB1000"
    root_posters = r"C:\Users\vankomme\datasets\IMDB1000\posters"
    csv_file_path = os.path.join(root_dataset, 'Top_1000_IMDB_movies.csv')
    json_file_path = os.path.join(root_dataset,'top_1000_imdb.json')
    json_db_path = os.path.join(root_dataset,'imdb_records.json')
    dataset_path_corpus= os.path.join(root_dataset, 'imdb_corpus.pkl')

    # api_key is to be removed form the code base
    api_key = '7282a1f6'

    # the storage of the movie records
    imdb = {}

    def __init__(self):
        imdb= self.load_CSV_dataset()
        #self.get_posters(imdb)
        self.merge(imdb)

        # Load the BEIR/scifact dataset from file or from hugging face
        if not os.path.exists(self.dataset_path_corpus):
            #self.dataset_corpus = load_dataset("BeIR/scifact","corpus", split="corpus")
            # "corpus", split="corpus"                    
            print("File doesn't exist!")
            self.dataset_corpus = load_dataset('json', data_files= self.json_db_path, split='train')
            # remove duplicated entries
            deduplicated_dataset = self.deduplicate(self.dataset_corpus,'Movie Name')            
            # Inspect the dataset
            print(deduplicated_dataset)
            # Writing to a picke file
            with open(self.dataset_path_corpus, "wb") as file:  # 'wb' is for write-binary mode
                pickle.dump(deduplicated_dataset, file)                

        else:
            print('the dataset exists.')
            with open(self.dataset_path_corpus, "rb") as file:  # 'rb' is for read-binary mode
                self.dataset_corpus = pickle.load(file)

    # return the hugging face dataset object
    def load_dataset_corpus(self):
        return self.dataset_corpus

    # load from a CSV 1000 extract into a list of dictionaries
    def load_CSV_dataset (self):
        print('Load the IMDB ')
        # Read the CSV file and convert it to a dictionary
        with open(self.csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = [row for row in csv_reader]
        
        # Write the dictionary to a JSON file
        with open(self.json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)

        with open(self.json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data

    def get_posters_url_from_omdb(self,title):
        # Construct the URL to query the OMDb API
        url = f"http://www.omdbapi.com/?t={title}&apikey={self.api_key}"
        
        # Make the request to the OMDb API
        response = requests.get(url)
        movie_data = response.json()
        
        # Check if the response contains a valid poster URL
        if 'Poster' in movie_data and movie_data['Poster'] != 'N/A':
            poster_url = movie_data['Poster']
            return poster_url
        else : 
            print('Poster not found '+title)   
            return ""


    def dump_movie_posters(self,db):
        file_id = 0
        for d in db:
            title = d['Movie Name']
            print(f"{file_id}: title {title} a poster link {d['Poster_Link']}")
            self.get_movie_poster(d['Poster_Link'],file_id)
            file_id +=1


    # example from ChatGPT
    def get_movie_poster_url_from_imdb(title, api_key):
        # Construct the URL to query the OMDb API
        url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        
        # Make the request to the OMDb API
        response = requests.get(url)
        movie_data = response.json()
        
        # Check if the response contains a valid poster URL
        if 'Poster' in movie_data and movie_data['Poster'] != 'N/A':
            poster_url = movie_data['Poster']
            
            # Download the poster image
            poster_response = requests.get(poster_url)
            
            # Save the poster image to a file
            with open('movie_poster.jpg', 'wb') as file:
                file.write(poster_response.content)
            
            print("Poster downloaded successfully as 'movie_poster.jpg'")
        else:
            print("Poster not found for this movie.")



    def get_movie_poster_from_url (self, url, file_id):

        image_path = os.path.join(self.root_posters, f"{file_id}.jpg")
        if os.path.exists(image_path): return

        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open a file in binary write mode and save the content
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            #print(f"Image successfully downloaded as ")
        else:
            print(f"Failed to retrieve the image: '{image_path}' of {file_id}")

    '''
    Returs a list of all movie records
    '''
    def load(self):
        with open(self.json_db_path, 'r', encoding='utf-8') as json_file:
            record_list = json.load(json_file)
        return record_list        

    def get_poster(self, record_id):
        print('get the poster image')


    def deduplicate (self, dataset, entry_name):
        # Get unique values of the entry_name column
        unique_texts = set()
        unique_indices = []

        for idx, entry in enumerate(dataset):
            if entry[entry_name].strip() not in unique_texts:
                unique_texts.add(entry[entry_name])
                unique_indices.append(idx)
            else :
                print('Dupicate entry: '+ str(entry[entry_name] ))

        # Filter the dataset to keep only unique entries
        deduplicated_dataset = dataset.select(unique_indices) 
        return deduplicated_dataset  

    def get_posters(self,imdb):     
        id = 0
        for m in imdb :
            print(str(id))
            title = imdb[id]['Movie Name']
            print(title)
            url = self.get_posters_url_from_omdb(title)
            print(url)
            if url != '' :
                self.get_movie_poster_from_url (url, id)
            id +=1

    def merge(self,imdb):

        if os.path.exists(self.json_db_path): return

        # list all posters
        posters = [f for f in os.listdir(self.root_posters) if f.endswith('.jpg')]

        db = []
        for p in posters:
            new_record = imdb[int(p.replace(".jpg",""))]
            db.append(new_record)

        with open(self.json_db_path, mode='w', encoding='utf-8') as json_file:
            json.dump(db, json_file, indent=4) 


    def hf_extract(self) :

        def replace_key_first(d, old_key, new_key, new_value):
            # Create a new ordered dictionary
            new_dict = {new_key: new_value}
            for k, v in d.items():
                if k != old_key:
                    new_dict[k] = v
            return new_dict

        print("create an hugging face format")  

        root_dataset = r"C:\Users\vankomme\datasets\watch_lists\imdb"
        meta_file_name = os.path.join(root_dataset,"metadata.jsonl")
        print(meta_file_name)
        if os.path.exists(meta_file_name):
            os.remove(meta_file_name)    

        imdb = IMDB()
        dataset_corpus = imdb.load_dataset_corpus() 
        print(dataset_corpus.shape) 
        data = dataset_corpus.to_list()  # or list(dataset)

        with open(meta_file_name, "a", encoding="utf-8") as f:    
            for d in data:
                # print(f" first item: {d['']}") 
                value = f"{d['']}.jpg"
                d=replace_key_first(d,"", "file_name", value)
                json.dump(d, f)
                f.write("\n")            

        # test hf formating

        dataset_imdb = load_dataset(root_dataset)
        print(dataset_imdb.shape)
        print(f"first item {dataset_imdb['train'][0]}")         
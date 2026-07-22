'''
Created on 20 mars 2020

https://developers.google.com/custom-search/v1/overview (100 queries per day are free)
    
the JSON API doc and testing
https://developers.google.com/custom-search/v1/?apix=true

A typical URL
https://www.googleapis.com/customsearch/v1?key=YOUR_GOOGLE_CSE_KEY&cx=YOUR_GOOGLE_CSE_CX&q=lectures
    
Tuto: https://towardsdatascience.com/current-google-search-packages-using-python-3-7-a-simple-tutorial-3606e459e0d4


CHE-109.426.601 of ZPF+Ingenieure+AG is missing

@author: vankomme
'''


import os
import urllib.request, urllib.parse
import json
import time
import codecs
from os.path import sep, isfile
import logging as log

from dotenv import load_dotenv

# Load credentials from .env (api key + custom search engine id).
load_dotenv()
KEY = os.getenv("GOOGLE_CSE_KEY", "")
CX = os.getenv("GOOGLE_CSE_CX", "")

MAX_PATH =260 

    
def search(searchfor): # pragma: no cover
    '''
    Search with Google CSE 
    
    PARAMETERS
    ----------
    searchfor : string
            The string pattern with want to search for
        
    RETURNS
    -------
        json results of the search
    
    '''
    query = urllib.parse.urlencode({'q': searchfor})
    url = f'https://www.googleapis.com/customsearch/v1?key={KEY}&cx={CX}&{query}&gl=ch' 
    
    log.info(url)
    search_response = urllib.request.urlopen(url)
    search_results = search_response.read().decode("utf8")
    return json.loads(search_results)


def retrieve_search(query,dest_dir): # pragma: no cover
    '''
    retrieve a previous search with Google CSE or make a new one 
    
    PARAMETERS
    ----------
    query : list uid and company_name
            The search pattern with want to query
    dest_dir : string
            The file path where the json file will be stored
        
    RETURNS
    -------
        json results of the search
    
    '''
    
    uid = query[0]
    company_name = query[1] 
    file_path= dest_dir +sep+ uid +".json"
    wait_count =8 
    
    # test if the file exists
    if isfile(file_path) :
        # yes => we get from the file
        f = codecs.open(file_path, "r", "utf-8")
        search_results =f.read()
        f.close()
        return json.loads(search_results)
        
    else :
        time.sleep(10) # make a request every 10 seconds
        # no => we search with Google's CSE¨
        for _i in range(10) :
            try: 
                results = search(company_name)
                file = codecs.open(file_path, "w", "utf-8")
                file.write(json.dumps(results, indent=4, sort_keys=True))
                file.close()
                return results
            
            except Exception as e:
                # to pass Google throttling algorithm
                log.info("Waiting : Error %s --> %s",60*wait_count,e) 
                time.sleep(60*wait_count) # wait 4min, 8min, 16min, ...
                wait_count *=2
                  

        


def get_links(searchfor):  # pragma: no cover
    '''
    retrieve the list of link found by Google CSE 
    
    PARAMETERS
    ----------
    searchfor : string
        The string pattern with want to search for
        
    RETURNS
    -------
        The list of URL's string
    
    '''
    
    link_list = []
    results= retrieve_search(searchfor)
    #log.info(json.dumps(results, indent=4, sort_keys=True))
    for key in results :
        if key == "items" :
            item = results[key]
            for k in item :
                for f in k :
                    #log.info("field %s",f)
                    if f == "displayLink" : # could also be a candidate?
                        link_list.append(k[f])
                    if f == "formattedUrl" :
                        #log.info("FormattedUrl: %s",k[f])
                        pass
    
    # remove duplicates while preserving order
    seen = set()
    seen_add = seen.add
    return [x for x in link_list if not (x in seen or seen_add(x))]


#def get_snippet()


def get_public_url(searchfor): # pragma: no cover
    
    query = urllib.parse.urlencode({'q': searchfor})
    public_url = f'https://cse.google.com/cse?cx={CX}&{query}&gl=ch'

    #log.info(public_url)    

    return public_url

'''
Created on 20 mars 2020

Code sourced from
https://github.com/Azure-Samples/cognitive-services-REST-api-samples/blob/master/python/Search/BingWebSearchv7.py

@author: vankomme
'''

import os
import requests
import sys
import time
import codecs
import json
from os.path import sep, isfile
import logging as log

from dotenv import load_dotenv

r'''
classdocs
                                                        (1000 transactions free per month)
https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/quickstarts/python
https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference#promote
    
'''

# Load the Bing subscription key from .env.
load_dotenv()
subscription_key = os.getenv("BING_SEARCH_KEY", "")
assert subscription_key
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
#search_term = "Softcom technologies"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
      
        
def search(query): # pragma: no cover
    '''
    Search with Bing and returns  
    
    PARAMETERS
    ----------
    query :  string
            The search pattern with want to query

        
    RETURNS
    -------
        json results of bing search a json formated response
    
    '''  
    
    params = {
        "q": query, 
        'mkt': 'CH',
        'count' : '50'
        } 

    r'''
    , 
    'responseFilter' : 'webpages,-images,-news,-video', 
    'answerCount': '1', 
    'setlang' : 'en-US'}
    '''
           
    # Call the API
    try:
        response = requests.get(search_url, headers=headers, params=params)
        log.info(response.url)
        log.debug(response.headers)
        response.raise_for_status()
        return response.json()
    except Exception as ex:
        log.error("Error in search with Bing")
        raise ex
        sys.exit()
       


def retrieve_search(query,dest_dir): # pragma: no cover
    '''
    retrieve a previous search or make a new one 
    
    PARAMETERS
    ----------
    query : list of uid and company_name
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
    
    # test if the file exists
    if isfile(file_path) :
        # yes => we get from the file
        f = codecs.open(file_path, "r", "utf-8")
        search_results =f.read()
        f.close()
        return json.loads(search_results)
        
    else :
        time.sleep(5)
        # no => we search with Google's CSE
        results = search(company_name)  
        file = codecs.open(file_path, "w", "utf-8")
        file.write(json.dumps(results, indent=4, sort_keys=True))
        file.close()
        return results       
        
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

from connectors.credentials import (
    MissingCredential, get_credential, require_credential,
)

MAX_PATH =260

#: Results per API call. Ten is the CSE maximum, and the parameter exists so a
#: caller that wants breadth pages through with ``start`` rather than hoping
#: for a bigger page.
RESULTS_PER_PAGE = 10


def available():
    '''
    Whether Google CSE can be called at all: both credentials configured.

    Lets a caller offer the connector as one source among several and skip it —
    with a reason — instead of catching MissingCredential around every call.

    RETURNS
    -------
        True when GOOGLE_CSE_KEY and GOOGLE_CSE_CX are both set
    '''
    return bool(get_credential("GOOGLE_CSE_KEY") and get_credential("GOOGLE_CSE_CX"))


def search(searchfor, start=None, num=None): # pragma: no cover
    '''
    Search with Google CSE

    PARAMETERS
    ----------
    searchfor : string
            The string pattern with want to search for
    start : int
            1-based index of the first result to return, for paging beyond the
            first ten (CSE accepts up to 91)
    num : int
            How many results this call returns, at most RESULTS_PER_PAGE

    RETURNS
    -------
        json results of the search

    '''
    # Credentials (api key + custom search engine id) are read at call time so
    # the .env of the calling application is found wherever it was started.
    key = require_credential("GOOGLE_CSE_KEY")
    cx = require_credential("GOOGLE_CSE_CX")

    parameters = {'q': searchfor}
    if start:
        parameters['start'] = int(start)
    if num:
        parameters['num'] = min(int(num), RESULTS_PER_PAGE)
    query = urllib.parse.urlencode(parameters)
    url = f'https://www.googleapis.com/customsearch/v1?key={key}&cx={cx}&{query}&gl=ch'

    log.info(url)
    search_response = urllib.request.urlopen(url)
    search_results = search_response.read().decode("utf8")
    return json.loads(search_results)


def get_results(searchfor, pages=1): # pragma: no cover
    '''
    The results of a search as {title, url, snippet} dictionaries.

    What a caller building a list of sources wants: the address of each result
    with enough text to tell what it is. Paging is done here because one CSE
    call returns ten results and a set of entities rarely fits in ten.

    PARAMETERS
    ----------
    searchfor : string
            The string pattern with want to search for
    pages : int
            How many pages of RESULTS_PER_PAGE results to fetch

    RETURNS
    -------
        list of dicts with "title", "url" and "snippet", deduplicated on the
        URL, in result order; [] when the search answers nothing
    '''

    results = []
    seen = set()
    for page in range(max(1, int(pages))):
        start = page * RESULTS_PER_PAGE + 1
        try:
            answer = search(searchfor, start=start, num=RESULTS_PER_PAGE)
        except MissingCredential:
            # A configuration problem, not a transient one: let it out.
            raise
        except Exception as error:
            # A refused page (quota, throttling) keeps the pages already read.
            log.warning("Google CSE page %s failed: %s", start, error)
            break
        items = answer.get("items") or []
        for item in items:
            url = (item.get("link") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({
                "title": (item.get("title") or "").strip(),
                "url": url,
                "snippet": (item.get("snippet") or "").strip(),
            })
        if len(items) < RESULTS_PER_PAGE:
            break  # the last page: asking for the next one only wastes a query
    log.info("Google CSE returned %s result(s) for %r", len(results), searchfor)
    return results


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
            
            except MissingCredential:
                # a configuration problem: retrying it only wastes hours
                raise

            except Exception as e:
                # to pass Google throttling algorithm
                log.info("Waiting : Error %s --> %s",60*wait_count,e)
                time.sleep(60*wait_count) # wait 4min, 8min, 16min, ...
                wait_count *=2
                  

        


def get_links(searchfor, pages=1):  # pragma: no cover
    '''
    retrieve the list of link found by Google CSE

    PARAMETERS
    ----------
    searchfor : string
        The string pattern with want to search for
    pages : int
        How many pages of results to fetch

    RETURNS
    -------
        The list of URL's string, deduplicated, in result order

    NOTES
    -----
        Returns each result's "link" — the address of the page. It used to
        collect "displayLink", which is the host alone (www.arena.ch for
        www.arena.ch/fr/fribourg): enough to name a site, never enough to
        fetch the page that was found. It also called retrieve_search() with
        one argument against its two-argument signature, so it raised
        TypeError on every call.
    '''

    return [result["url"] for result in get_results(searchfor, pages=pages)]


#def get_snippet()


def get_public_url(searchfor): # pragma: no cover
    '''
    The address of the same search on Google's public CSE page.

    PARAMETERS
    ----------
    searchfor : string
        The string pattern with want to search for

    RETURNS
    -------
        the URL a human can open to see these results
    '''

    # The engine id used to be a module-level constant; it is a credential
    # read at call time since, and this line still named the constant.
    cx = require_credential("GOOGLE_CSE_CX")
    query = urllib.parse.urlencode({'q': searchfor})
    return f'https://cse.google.com/cse?cx={cx}&{query}&gl=ch'

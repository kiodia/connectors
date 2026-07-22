import os
import re
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging
from bs4 import BeautifulSoup
import subprocess
import traceback

from config import flags
# from connectors.swiss_enterprise import SwissEnterpriseDB


# Constants
FS = os.path.sep
TAB = "\t"
APPEND = True
ERROR_STATUS = 1
PAGE_SIZE = 100  # this should be read from the xml file
NB_RETRY = 3
WORK_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday
V = "v1"  # Version
ACTIVE = "active"
URL_ZEFIX_SEARCH = "https://www.zefix.ch/en/search/entity/list/"
URL_DIRECT = "?name="
DATASET_BOOTSTRAP = "fosc_" + V + "_20180903.tsv"

log = logging.getLogger() 

class F(Enum):
    uid = 0
    name = 1
    status = 2
    legal_seat = 3
    legal_form = 4
    register_office = 5
    address_street = 6
    address_city = 7
    link = 8
    purpose_language = 9
    purpose = 10
    publication_dates = 11

class RegMap(Enum):
    ZH = "ZH"
    BE = "BE"
    LU = "LU"
    UR = "UR"
    SZ = "SZ"
    OW = "OW"
    NW = "NW"
    GL = "GL"
    ZG = "ZG"
    FR = "FR"
    SO = "SO"
    BS = "BS"
    BL = "BL"
    SH = "SH"
    AR = "AR"
    AI = "AI"
    SG = "SG"
    GR = "GR"
    AG = "AG"
    TG = "TG"
    TI = "TI"
    VD = "VD"
    VS = "VS"
    NE = "NE"
    GE = "GE"
    JU = "JU"

    def get_reg(self):
        return self.value

class AmtsblattXML:
    '''

    Amtsblatt information on the API https://amtsblattportal.ch/docs/api/

    This module has been translated from Java code

    AmtsblattXML collect files from the portal and not bound to any specific DB or vector DB

    '''

    def __init__(self, dest_dir: str):
        self.dest_dir = dest_dir
        self.token = ""
        self.dataset = Dataset()
        # Initialize language detector (placeholder - would need actual implementation)
        self.detector = None  # LanguageDetectorBuilder.fromIsoCodes("de", "fr", "it").build()


    def ingest_amstblatt(self, start_date: datetime):
        '''
        Ingest the full amsblatt into a Vector DB

        Args:
            start_date: From when we uplaod the Amtsblatt publications

        '''
        try:  
            # start_date = datetime(year=2025, month=5, day=28)  # Septembre 20, 2025        
            self.retrieve_missing_bulk_xml_till_yesterday(start_date)
            
            # to regenerate remove the directory of the date
            self.retrieve_missing_publications_xml(start_date)
                    
        except Exception as e:
            message = f"AmstblattXML error {str(e)}"
            log.error(message)    



    # def login(self):
    #     '''
    #     Deprecated not needed any more

    #     '''

    #     cmd = "curl https://amtsblattportal.ch/api/v1/login --data username=robert.vankommer@gmail.com&password=vank@FOSC7 -i"
    #     log.info(cmd)

    #     try:
    #         process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         stdout, stderr = process.communicate()

    #         for line in stdout.decode('utf-8').splitlines():
    #             log.debug(line)
    #             if line.startswith("x-auth-token:"):
    #                 self.token = line.split(" ")[1].strip()
    #                 log.info(f"login token: {self.token}")

    #         exit_code = process.wait()
    #         if exit_code != 0:
    #             message = f"Error in login with exit code: {exit_code}"
    #             log.error(message)
    #             self._log_error_and_terminate(message)

    #         log.info("Terminates login process.")
    #         time.sleep(1)  # wait one second

    #     except Exception as e:
    #         message = f"login failed in amtsblattportal: {str(e)}"
    #         log.error(message)
    #         # self._log_error_and_terminate(message)

    def get_counter_bulk_file(self, file_path: str) -> List[int]:
        lines = Utils().read_utf8_list(file_path)
        counters = [0, 0, 0]

        for line in lines:
            if "<subRubric>HR01" in line:
                counters[0] += 1
            if "<subRubric>HR02" in line:
                counters[1] += 1
            if "<subRubric>HR03" in line:
                counters[2] += 1

        return counters

    def show_uid(self, uid: str, data: Dict[str, str]):
        log.info("-----------------------------------------------")
        line = data.get(uid)
        if not line:
            log.info(f"no data available for this {uid}")
            return

        fields = line.split(TAB)
        for field in F:
            log.info(f"{field.name}= {fields[field.value]}")

    def x_check(self):
        pattern = re.compile(r"\d{8}_bulk.xml")
        self._purge_bulks(Path(self.dest_dir), pattern)

    def update_day(self, day_date:datetime, fosc):
  
        dir_path = os.path.join(flags['dest_dir'],day_date.strftime("%Y"),"xml","publications",day_date.strftime("%Y%m%d"))

        if not os.path.exists(dir_path):
            log.info(f"No publications are available for this day {day_date.strftime("%Y%m%d")} in {dir_path}")
            return

        publications_list = os.listdir(dir_path) 
        log.info(f"Number of publication {len(publications_list)} in {dir_path}")

        for file in publications_list :
            file_path = os.path.join(dir_path, file)

            log.debug(f" update_day of {file_path}")

            """Process a single XML file"""
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    xml_content = file.read()

                # Parse XML with BeautifulSoup
                try:
                    soup = BeautifulSoup(xml_content, 'xml')
                except Exception as e:
                    log.error(f"XML parsing error: {e} in file {file_path}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Find all subRubric elements
                subrubric_text = soup.find('subRubric').text

                log.debug(f"subrubric  {subrubric_text}")

                if subrubric_text == 'HR01':
                    fosc.add_new_entry(soup)
                elif subrubric_text == 'HR02':
                    fosc.mutate_entry(soup)
                elif subrubric_text == 'HR03':
                    fosc.delete_entry(soup)
                    
            except Exception as e:
                log.error(f"Error processing file {file_path}: {e}")
                size = os.path.getsize(file_path)
                log.error(f"The file size is: {size}")
                if size == 260 : 
                    os.remove(file_path)
                    log.error(f"The file has been removed")
                continue
                # traceback.print_exc()
                # sys.exit(1)
                


    def retrieve_missing_bulk_xml_till_yesterday(self, start_date: datetime):
        yesterday = datetime.now() - timedelta(days=1)
        self.retrieve_missing_bulk_xml(start_date, yesterday)

    def retrieve_missing_bulk_xml(self, start_date: datetime, end_date: datetime):
        current_date = start_date
        date_validator = DateValidator()

        while current_date <= end_date:
            if date_validator.valid_publication_day(current_date):
                if self._new_xml(current_date):
                    # the new xml bulk file doesn't exist yet => get it
                    log.info(f"get the xml at date: {current_date}")
                    try:
                        self._get_bulk_xml(current_date)
                    except Exception as e:
                        log.error(str(e))
                        #self._log_error(str(e))
            current_date += timedelta(days=1)

    def retrieve_missing_publications_xml(self, start_date: datetime):
        current_date = start_date
        yesterday = datetime.now() - timedelta(days=1)
        date_validator = DateValidator(WORK_DAYS)

        while current_date <= yesterday:
            if date_validator.valid_publication_day(current_date):
                log.info(f"retrieve pubs at {current_date}")
                year = current_date.year
                date_str = current_date.strftime("%Y%m%d")
                dir_path = os.path.join(self.dest_dir, str(year), "xml", "publications", date_str)
                
                publication_ids = self._parse_bulk(current_date)
                if not publication_ids:
                    current_date += timedelta(days=1)
                    continue

                if not os.path.exists(dir_path):
                    log.info(f"new dir {dir_path} nb pubs found: {len(publication_ids)}")
                    os.makedirs(dir_path)
                    for pub_id in publication_ids:
                        file_path = os.path.join(dir_path, f"{pub_id}.xml")
                        log.info(file_path)
                        try:
                            self._get_publication(pub_id, file_path)
                        except Exception as e:
                            log.error(f"Error getting publication: {e}")
                else:
                    # get the existing publication_ids in the directory
                    existing_files = set(f.replace(".xml", "") for f in os.listdir(dir_path) if f.endswith(".xml"))
                    missing_ids = set(publication_ids) - existing_files
                    for pub_id in missing_ids:
                        file_path = os.path.join(dir_path, f"{pub_id}.xml")
                        log.info(f"Get the missing publication xml file: {file_path}")
                        try:
                            self._get_publication(pub_id, file_path)
                        except Exception as e:
                            log.error(f"Error getting missing publication: {e}")
            current_date += timedelta(days=1)

    def test_uid_read(self):
        cmd = f"curl -X GET https://example.com/api -H 'x-auth-token: {self.token}'"
        log.info(cmd)
        file_path = os.path.join(self.dest_dir, "test.xml")

        try:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(file_path, 'a', encoding='utf-8') as output_file:
                for line in process.stdout:
                    line = line.decode('utf-8').strip()
                    log.info(line)
                    if "<total>" in line:
                        total = line[line.index(">")+1:line.rindex("<")].strip()
                        log.info(total)
                    output_file.write(line + "\n")

            exit_code = process.wait()
            if exit_code != 0:
                message = f"Exit code error from launchCurlProcess process: {file_path}"
                log.error(message)
                self._log_error_and_terminate(message)

        except Exception as e:
            message = f"Internal Error in launchCurlProcess {str(e)}"
            log.error(message)
            self._log_error_and_terminate(message)

    # Private methods
    def _to_be_saved(self, path: str) -> bool:
        basename = os.path.basename(path)
        yyyy = basename.split("_")[-1][:4]
        mm = basename.split("_")[-1][4:6]
        backup_dir = os.path.join(self.dest_dir, yyyy, "backup")

        log.info(f"backupDir {backup_dir}")

        if not os.path.exists(backup_dir):
            return True

        for filename in os.listdir(backup_dir):
            if filename.startswith(f"fosc_{V}_") and filename.endswith(".tsv"):
                date_part = filename.split("_")[-1][:6]
                if date_part == yyyy + mm:
                    return False
        return True

    def _move_it(self, path: str):
        basename = os.path.basename(path)
        yyyy = basename.split("_")[-1][:4]
        backup_path = path.replace("fosc_", os.path.join(yyyy, "backup", "fosc_"))
        log.info(f"Dataset moved from {path} --> {backup_path}")

        try:
            os.renames(path, backup_path)
        except OSError as e:
            log.error(f"{str(e)} by moving dataset {path}")

    def _purge_bulks(self, directory: Path, pattern: re.Pattern):
        for entry in directory.iterdir():
            if entry.is_file() and pattern.match(entry.name):
                log.debug(f"File {entry}")
                if entry.stat().st_size > 200:
                    counters = self.get_counter_bulk_file(str(entry))
                    total = counters[0] + counters[1] + counters[2]
                    pdf_file = str(entry).replace("xml", "html").replace("_bulk", "")
                    pdf_content = PDFContent(self.dest_dir).get_pdf_summary_content(pdf_file)
                    if len(pdf_content) != total:
                        log.info(f"# of entries {len(pdf_content)} = {total}")
                        log.info(f"File corrupted: {entry}")
            elif entry.is_dir():
                self._purge_bulks(entry, pattern)

    # def _add_new_one(self, soup):

    #     name = soup.find('company').find('name').text
    #     log.info(f"New company {name}")

        # publications = soup.find_all('publication')
        # if total == str(len(publications)):
        #     for pub in publications:
        #         if pub.find('subrubric').text == "HR01":
        #             record = self._get_dataset_record(pub, bulk_file)
        #             if record:
        #                 data[record[F.uid.value]] = Dataset.build_line(record)
        #                 log.debug(f"Adds the new one: {record[F.uid.value]}")
        # else:
        #     log.error(f"The number of bulk publications does not correspond to the total expected: {total}")

    # def _mutation_entry(self, soup):

    #     name = soup.find('commonsNew').find('company').find('name').text
    #     log.info(f"Mutation of the company {name}")

        # publications = soup.find_all('publication')
        # if total == str(len(publications)):
        #     for pub in publications:
        #         if pub.find('subrubric').text == "HR02":
        #             record = self._get_dataset_record(pub, bulk_file)
        #             if record:
        #                 uid = record[F.uid.value]
        #                 old_data = data.get(uid)
        #                 if old_data:
        #                     old_fields = old_data.split(TAB)
        #                     pub_dates = f"{old_fields[F.publication_dates.value]} {record[F.publication_dates.value]}"
        #                     record[F.publication_dates.value] = pub_dates.strip()
        #                     log.debug(f"Mutated: {uid}")
        #                     data.pop(uid)
        #                 else:
        #                     log.debug(f"On mutation, there was no previous entry to mutate for {uid}")
        #                 data[record[F.uid.value]] = Dataset.build_line(record)
        # else:
        #     log.error(f"The number of bulk publications does not correspond to the total expected: {total}")

    # def _deletion_entry(self, soup):

    #     name = soup.find('company').find('name').text
    #     log.info(f"Deletion of the company name {name}")

        # publications = soup.find_all('publication')
        # if total == str(len(publications)):
        #     for pub in publications:
        #         if pub.find('subrubric').text == "HR03":
        #             pub_id = pub.find('id').text
        #             bulk_date = os.path.basename(bulk_file).split('_')[0]
        #             file_path = self._make_file_path(bulk_date, pub_id)
        #             log.debug(f"deletion {file_path}")
                    
        #             record = Utils.read_utf8(file_path)
        #             rec_soup = BeautifulSoup(record, 'xml')
        #             uid = self._format_uid(rec_soup.find('uid').text)
                    
        #             if uid in data:
        #                 data.pop(uid)
        #                 log.debug(f"Removed: {uid}")
        #             else:
        #                 log.debug(f"On deletion, there was no previous entry to be removed for: {uid}")
        # else:
        #     log.error(f"The number of bulk publications does not correspond to the total expected: {total}")

    # def _get_dataset_record(self, pub_element, bulk_file: str) -> List[str]:
    #     # pub_id = pub_element.find('id').text
    #     # bulk_date = os.path.basename(bulk_file).split('_')[0]
    #     # file_path = self._make_file_path(bulk_date, pub_id)
    #     # log.debug(file_path)
        
    #     # record = Utils.read_utf8(file_path)
    #     # soup = BeautifulSoup(record, 'xml')
        
    #     record = Dataset.init_line(len(F))
    #     uid = self._format_uid(soup.find('uid').text)
    #     record[F.uid.value] = uid
    #     record[F.name.value] = soup.find('name').text
    #     record[F.publication_dates.value] = soup.find('publicationdate').text.replace("-", "")
    #     record[F.legal_seat.value] = soup.find('seat').text
    #     record[F.status.value] = ACTIVE
    #     record[F.register_office.value] = self._map_register(soup.find('officename').text)
    #     record[F.legal_form.value] = soup.find('legalform').text
    #     record[F.link.value] = f"{URL_ZEFIX_SEARCH}{uid.replace('-', '').replace('.', '')}{URL_DIRECT}"
    #     record[F.purpose.value] = soup.find('purpose').text
        
    #     # Language detection would go here
    #     # Placeholder - would need actual implementation
    #     record[F.purpose_language.value] = "de"  # Default
        
    #     if soup.find('noaddress').text == "false":
    #         number = soup.find('housenumber').text
    #         record[F.address_street.value] = f"{soup.find('street').text} {number}"
    #         zip_code = soup.find('swisszipcode').text
    #         record[F.address_city.value] = f"{zip_code} {soup.find('town').text}"
        
    #     return record

    def _map_register(self, office_name: str) -> str:
        clean_name = office_name.replace("-", "").replace(".", "").replace(" ", "")
        for reg in RegMap:
            if reg.name in clean_name:
                return reg.get_reg()
        log.error(f"No register found for {office_name}")
        sys.exit(ERROR_STATUS)
        return ""

    def _format_uid(self, uid: str) -> str:
        if "." in uid:
            return uid
        return f"CHE-{uid[3:6]}.{uid[6:9]}.{uid[9:12]}"

    def _make_file_path(self, bulk_date: str, pub_id: str) -> str:
        year = bulk_date[:4]
        return os.path.join(self.dest_dir, year, "xml", "publications", bulk_date, f"{pub_id}.xml")

    def _get_bulk_files(self, latest_dataset: str) -> 'BulkFiles':
        bulk_files = BulkFiles()
        bulk_files.current_dataset = latest_dataset
        
        try:
            date_str = os.path.basename(latest_dataset).split('_')[-1].split('.')[0]
            date = datetime.strptime(date_str, "%Y%m%d")
            year = date.year
            date_str = date.strftime("%Y%m%d")
            
            bulk_files.current_bulk_file = os.path.join(
                self.dest_dir, str(year), "xml", f"{date_str}_bulk.xml")
            bulk_files.next_dataset = self._get_next_dataset_file(bulk_files)
        except Exception as e:
            message = f"Error occurred in getBulkFiles {str(e)}"
            log.error(message)
            self._log_error(message)
        
        return bulk_files

    def _get_next_dataset_file(self, bulk_files: 'BulkFiles') -> str:
        years = self._get_years()
        bulk_files_list = []
        
        for year in years:
            dir_path = os.path.join(self.dest_dir, year, "xml")
            if os.path.exists(dir_path):
                for entry in os.listdir(dir_path):
                    if entry.endswith("_bulk.xml"):
                        bulk_files_list.append(os.path.join(dir_path, entry))
        
        bulk_files_list.sort()
        
        try:
            current_index = bulk_files_list.index(bulk_files.current_bulk_file)
            if current_index + 1 < len(bulk_files_list):
                next_bulk = bulk_files_list[current_index + 1]
                date_str = os.path.basename(next_bulk).split('_')[0]
                return os.path.join(self.dest_dir, f"fosc_{V}_{date_str}.tsv")
        except ValueError:
            pass
        
        return ""

    def _get_latest_dataset(self) -> str:
        datasets = []
        
        for entry in os.listdir(self.dest_dir):
            if entry.endswith(".tsv") and f"fosc_{V}_" in entry:
                datasets.append(os.path.join(self.dest_dir, entry))
        
        if datasets:
            datasets.sort()
            return datasets[-1]
        return ""

    def _clean_up_bulks(self, file_path: str, page_count: 'PageCount'):
        lines_to_skip = 7
        skip = 0
        # The page are counted as nbpage-1 to 0
        # with 1823 publication we have 18 pages
        # with 1800 we read one too much that doesn't exist (=> we don't care)
        nb_pages = page_count.total // PAGE_SIZE
        
        with open(file_path, 'r', encoding='utf-8') as input_file:
            with open(file_path + ".saved", 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    if "</bulk:bulk-export>" in line and nb_pages > 0:
                        skip = lines_to_skip
                        nb_pages -= 1
                    
                    if skip > 0:
                        skip -= 1
                    else:
                        output_file.write(line)
        
        os.remove(file_path)
        os.rename(file_path + ".saved", file_path)

    def _parse_bulk(self, date: datetime) -> List[str]:
        year = date.year
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(self.dest_dir, str(year), "xml", f"{date_str}_bulk.xml")
        log.info(f" The bulk file to process {file_path}")
        
        if not os.path.exists(file_path):
            return []
        
        xml = Utils.read_utf8(file_path)
        # pip install lxml is needed
        soup = BeautifulSoup(xml, 'xml')
        total_tag = soup.find('total')
        if total_tag:
            total = total_tag.text
            log.info(f"Total number of publication expected: {total}")
        else :
           log.error(f" no total xml tag found in {file_path}")
           return
        
        publications = []
        for meta in soup.find_all('meta'):
            if meta.find('rubric').text == "HR":
                publications.append(meta.find('id').text)
        return publications

    def _launch_request(self, url: str, file_path: str, page: int) -> 'PageCount':
        '''
        Launch a page read request to amstblatt portal

        Args:
            url: the url of amstblatt
            file_path: the file where the pages are appended
            page: the number of the page counted from 1 to nbpages

        Returns:
            PageCount: the updated page counter
        
        '''
        page_count = PageCount()
        log.info(url)
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers= headers)

            # print(response.status_code)
            # print(f" file location {file_path}")
            # print(response.request.headers)
            # print(response.text[:500])
            
            if response.status_code == 200:
                with open(file_path, 'a', encoding='utf-8') as output_file:
                    for line in response.text.splitlines():
                        # each pages contains the total publication info
                        if "<total>" in line:
                            total = line[line.index(">")+1:line.rindex("<")].strip()
                            # the page counting should be rewritten
                            page_count.total = int(total) # for instance 1823 or 1800
                            page_count.remaining = page_count.total - PAGE_SIZE * page # 23 for 1823 and 0 for 1800
                            if page_count.remaining < 0:
                                page_count.remaining = 0
                                log.error(f"The remaining page_count {page_count.remaining} is negative, current page {page} from {total}")
                        output_file.write(line + "\n")
            else:
                log.error(f"Exit code error= {response.status_code}, from request: {file_path}")
                sys.exit(ERROR_STATUS)
        except Exception as e:
            log.error(f"Error in requesting {url} with the error {str(e)}")
        
        return page_count

    def _get_publication(self, pub_id: str, file_path: str):

        # https://amtsblattportal.ch/api/v1/publications/2f08b537-dc78-4ad8-9ccd-afb5f80fe22d/xml

        url = f"https://amtsblattportal.ch/api/v1/publications/{pub_id}/xml"  # Actual URL
        log.debug(url)
        
        for attempt in range(NB_RETRY + 1):
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code == 200:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    return
                elif attempt < NB_RETRY:
                    wait_time = 60 + 2 * attempt * 60
                    log.info(f"{attempt + 1} retry to read {file_path} publication id {pub_id}")
                    time.sleep(wait_time)
            except Exception as e:
                if attempt == NB_RETRY:
                    message = f"HTTP error in getPublication: {response.status_code if 'response' in locals() else 'No response'} from getPublication process: {pub_id} {file_path}"
                    log.error(message)
                    #self._log_error(message)
                continue

    def _get_bulk_xml(self, date: datetime):

        # https://amtsblattportal.ch/api/v1/publications/xml?
        # publicationStates=PUBLISHED
        # &rubrics=KK&
        # &publicationDate.start=2017-01-01
        # &publicationDate.end=2017-01-01

        date_str = date.strftime("%Y-%m-%d")
        base_url = f"https://amtsblattportal.ch/api/v1/publications/xml?publicationStates=PUBLISHED&publicationDate.start="
        url = f"{base_url}{date_str}&publicationDate.end={date_str}"  # get bulk pages for one day
        log.debug(url)
        
        year = date.year
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(self.dest_dir, str(year), "xml", f"{date_str}_bulk.xml")
        # let's create the directory of the path
        dir_path = os.path.dirname(file_path)   # get base directory
        os.makedirs(dir_path, exist_ok=True)
        log.info(file_path)
        
        # request page 0
        page_count = self._launch_request(url, file_path, 0)

        page = 0        
        while page_count.remaining != 0: # if we still have pages left in a last page
            page += 1
            # the page are counted from 1 to nbpages            
            next_url = f"{url}&page={page}"
            log.debug(f"get new page {page}")

            # update page_count.remaining, the pages are counted from 1 to nbpages
            page_count = self._launch_request(next_url, file_path, page)
        
        self._clean_up_bulks(file_path, page_count)

    def _new_xml(self, date: datetime) -> bool:
        year = date.year
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(self.dest_dir, str(year), "xml", f"{date_str}_bulk.xml")
        log.info(file_path)
        return not os.path.exists(file_path)

    def _get_years(self) -> List[str]:
        current_year = datetime.now().year
        return [str(year) for year in range(2018, current_year + 1)]

    def _log_error(self, message: str):
        # Placeholder for error logging implementation
        pass

    def _log_error_and_terminate(self, message: str):
        self._log_error(message)
        sys.exit(1)

class BulkFiles:
    def __init__(self):
        self.current_bulk_file = ""
        self.current_dataset = ""
        self.next_dataset = ""

class PageCount:
    def __init__(self):
        self.remaining = 0
        self.total = 0

class Dataset:
    @staticmethod
    def init_line(length: int) -> List[str]:
        return [""] * length

    @staticmethod
    def build_line(fields: List[str]) -> str:
        return TAB.join(fields)

    @staticmethod
    def read(file_path: str) -> Dict[str, str]:
        data = {}
        if not os.path.exists(file_path):
            return data
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split(TAB)
                if fields:
                    data[fields[F.uid.value]] = line.strip()
        return data

    @staticmethod
    def write(data: Dict[str, str], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in data.values():
                f.write(line + "\n")

class Utils:
    def read_utf8(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            log.error(f"Failed to read UTF-8 file: {file_path} {str(e)}")
            return ""


    def read_utf8_list(file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        except Exception as e:
            log.error(f"Failed to read UTF-8 file: {file_path} {str(e)}")
            return []

class DateValidator:
    def __init__(self, work_days=None):
        self.work_days = work_days or [0, 1, 2, 3, 4]  # Monday to Friday by default

    def valid_publication_day(self, date: datetime) -> bool:
        return date.weekday() in self.work_days

class PDFContent:
    def __init__(self, dest_dir: str):
        self.dest_dir = dest_dir

    def get_pdf_summary_content(self, pdf_file: str) -> List[str]:
        # Placeholder for PDF content extraction
        return []

# Example usage
# if __name__ == "__main__":
#     connector = AmtsblattXML("/path/to/dataset")
#     connector.login()
#     connector.update()

import logging

log = logging.getLogger()

class Bootstrap:
    '''
    Extense the bootstrop dataset with two empty fields web_link and web_summary

    # Load dataset
    dataset = load_dataset("csv", data_files="data.tsv", delimiter="\t", features=features)

    '''

    def __init__(self, source_file_path, destination_file_path):
        '''
        Init the ingest a HF bootstrap dataset to Qdrant.

        Args:
            source_file_path: The file location of the HF bootstrap dataset
            destinatiion_file_path: The extended dataset 
        '''
        self.source_file_path = source_file_path
        self.destination_file_path = destination_file_path
        log.info(f"Bootstrap source is extended to match the size of hf data features.")

    def get_hf_fosc(self, extended_nb_fields, creation_date):
        '''
        Add two empty fields to the source tsv dataset web_link, web_summary

        '''

        with open(self.source_file_path, "r", encoding="utf-8") as infile, \
            open(self.destination_file_path, "w", encoding="utf-8") as outfile:

            tabs = "\t" * (extended_nb_fields-1)
        
            for line in infile:
                line = line.rstrip("\n")  # remove newline
                if line.strip():          # skip completely empty lines
                    # Add two empty tab-separated fields at the end
                    line += tabs
                    line += "\t"+creation_date
                outfile.write(line + "\n")
       

        log.info(f"{self.destination_file_path} is created.")

    def get_hf_arix(self):
        pass

    def get_hf_midjourney_lib(self):
        pass
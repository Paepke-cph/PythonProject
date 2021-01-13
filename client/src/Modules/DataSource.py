import requests
import pandas as pd
import json

import Cleaner
def fetch(url, csv_file):
    try:
        re = requests.get(url)
        dataset = pd.read_json(re.text)
        if(len(dataset) > 0):
            # Swaps rows to columns
            dataset = dataset.T
            dataset.to_csv(csv_file)
            return Cleaner.clean_df(dataset)
        else:
            return Cleaner.get_and_clean_df(csv_file)
    except:
        dataset = Cleaner.get_and_clean_df(csv_file) 
        return dataset
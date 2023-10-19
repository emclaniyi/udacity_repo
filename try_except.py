import pandas as pd 
import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"SUCCESS: There are {df.shape} rows in your dataframe")
        logging.info("SUCCESS: your file was successfully read in.")
        return df
    except FileNotFoundError:
        logging.error('ERROR: we are unable to find the file')

df = read_data('some_path')
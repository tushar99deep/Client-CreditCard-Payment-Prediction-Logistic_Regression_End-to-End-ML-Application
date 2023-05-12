import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import Data_Transformation



#1st step - initialize data ingestion 

@dataclass
class Data_Ingestion_config:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

#create class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=Data_Ingestion_config()


    def initiate_data_ingestion(self):
        logging.info('Data Ingestion started')
        try:
            df=pd.read_csv(os.path.join('notebook/data/credit_card.csv'))
            logging.info('Data is read as pandas dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Train test split')

           
            train_set,test_set = train_test_split(df,test_size=0.7,random_state=142)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at data ingestion stage')
            raise CustomException(e,sys)





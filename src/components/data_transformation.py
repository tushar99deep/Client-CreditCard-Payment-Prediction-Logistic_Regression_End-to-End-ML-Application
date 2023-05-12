import sys

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class Data_Transformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            preprocessor= StandardScaler()
            logging.info('Preprocessor object created')

            return preprocessor


        except Exception as e:
            logging.error('Error in creating preprocessor object', e)
            raise CustomException(e,sys)
        
    


    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df.rename(columns={'Unnamed: 0':'Useless'},inplace=True)

            test_df.rename(columns={'Unnamed: 0':'Useless'},inplace=True)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation_object()

            
            #target_column_name = 'Y'
            #drop_columns = [target_column_name,'Useless']

            input_feature_train_df = train_df.drop(columns=['Useless','Y'],axis=1)
            target_feature_train_df = train_df[['Y']]

            input_feature_test_df = test_df.drop(columns=['Useless','Y'],axis=1)
            target_feature_test_df = test_df[['Y']]

            logging.info(f'input_feature_tarin_df : \n{input_feature_train_df.head().to_string()}')
            logging.info(f'target_feature_train_df  : \n{target_feature_train_df.head().to_string()}')
            logging.info(f'input_feature_test_df : \n{input_feature_test_df.head().to_string()}')
            logging.info(f'target_feature_test_df  : \n{target_feature_test_df.head().to_string()}')

            

            #train_arr = np.c_[np.array(input_feature_train_df), np.array(target_feature_train_df)]
            #test_arr = np.c_[np.array(input_feature_test_df), np.array(target_feature_test_df)]
             # Placeholder object to be saved


            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            logging.info('Error occured at initiate_data_transformation stage')
            raise CustomException(e,sys)


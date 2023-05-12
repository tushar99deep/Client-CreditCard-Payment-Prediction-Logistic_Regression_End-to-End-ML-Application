# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.utils import save_object


from dataclasses import dataclass
import sys
import os


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info('Training Logistic Regression Model')
            model= LogisticRegression(max_iter=10000,C= 0.001,  penalty= 'l2', solver= 'saga',fit_intercept= True,class_weight='balanced')
           
            
            logging.info('Model object created and best parameters are passed for model building')


            model.fit(X_train,y_train)
            y_pred= model.predict(X_test)
            logging.info('Model Training Complete')
            test_score = accuracy_score(y_pred,y_test)
            

            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=model
            )
            
        

        except Exception as e:
            logging.info('Error occured at model training stage')
            raise CustomException(e,sys)
        

'''
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = Data_Transformation()
    train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)
    '''         



            

            

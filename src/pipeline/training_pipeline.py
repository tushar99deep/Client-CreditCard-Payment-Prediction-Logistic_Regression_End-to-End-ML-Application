from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import ModelTrainer

import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd



if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = Data_Transformation()
    train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
 






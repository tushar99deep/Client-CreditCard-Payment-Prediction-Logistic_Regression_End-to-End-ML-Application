import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor= load_object(preprocessor_path)

            f=preprocessor.transform(features)

            
            model=load_object(model_path)

            
            
            pred=model.predict(f)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Customer_ID:int,X1:int,X2:int,X3:int,X4:int,X5:int,X6:int,X7:int,X8:int,X9:int,X10:int,X11:int,X12:int,
                 X13:int,X14:int,X15:int,X16:int, X17:int,X18:int,X19:int,X20:int,X21:int,X22:int,X23:int):
        self.Customer_ID=Customer_ID
        self.X1=X1
        self.X2=X2
        self.X3=X3
        self.X4=X4
        self.X5=X5
        self.X6=X6
        self.X7=X7
        self.X8=X8
        self.X9=X9
        self.X10=X10
        self.X11=X11
        self.X12=X12
        self.X13=X13
        self.X14=X14
        self.X15=X15
        self.X16=X16
        self.X17=X17
        self.X18=X18
        self.X19=X19
        self.X20=X20
        self.X21=X21
        self.X22=X22
        self.X23=X23
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Customer_ID':[self.Customer_ID],
                'X1':[self.X1],
                'X2':[self.X2],
                'X3':[self.X3],
                'X4':[self.X4],
                'X5':[self.X5],
                'X6':[self.X6],
                'X7':[self.X7],
                'X8':[self.X8],
                'X9':[self.X9],
                'X10':[self.X10],
                'X11':[self.X11],
                'X12':[self.X12],
                'X13':[self.X13],
                'X14':[self.X14],
                'X15':[self.X15],
                'X16':[self.X16],
                'X17':[self.X17],
                'X18':[self.X18],
                'X19':[self.X19],
                'X20':[self.X20],
                'X21':[self.X21],
                'X22':[self.X22],
                'X23':[self.X23],
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)



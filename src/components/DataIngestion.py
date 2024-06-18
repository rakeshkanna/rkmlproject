import os
import sys
from src.Exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.DataTransformation import DataTransformation,DataTransformationConfig
from src.components.ModelTrainer import Model_Trainer_Config,Model_Trainer
#Create Training Data class
@dataclass
class DataIngestionConfig:
    #train data path
    train_data_path = os.path.join('artifacts','train.csv')
    #Test Data Path
    test_data_path = os.path.join('artifacts','test.csv')
    #Raw Data Path
    raw_data_path = os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def InitiateDataIngestion_csv(self,file_path:str):
        logging.info("Starting data Ingestion")
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Created the dataframe from csv at {file_path}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f"Raw Data saved at {self.ingestion_config.raw_data_path}")
            train_set,test_set = train_test_split(df,test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info(f"Train Data saved at {self.ingestion_config.train_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Test Data saved at {self.ingestion_config.test_data_path}")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.InitiateDataIngestion_csv('./src/notebooks/data.csv')
    data_transformer = DataTransformation()
    train_arr, test_arr,_ = data_transformer.Initiate_Transform_Data(train_path=train_data,test_path= test_data)
    model_trainer=Model_Trainer()
    print(model_trainer.Train_Model(train_array=train_arr,test_array=test_arr))


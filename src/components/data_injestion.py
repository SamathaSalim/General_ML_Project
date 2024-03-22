import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
#input to the datainjestion component(initial configuration), 
#it now knows where to save the train,test,raw data after reading
#dataclass is used since we need to define variables only
@dataclass
class DataInjestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


#data injestion, reading data from a source
class DataInjestion:
    def __init__(self): #this contains above 3 variables (path)
        self.injestion_config=DataInjestionConfig()

    def initiate_data_injestion(self):
        logging.info("entered into the data injestion component")
        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("read dataset as dataframe")

            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)#creating afolder

            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)#read to raw_data_path

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.injestion_config.train_data_path,index=False,header=True)#read to train_data_path
            test_set.to_csv(self.injestion_config.test_data_path,index=False,header=True)#read to test_data_path

            logging.info("data injestion completed")
            
            return(
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=DataInjestion()
    train_data,test_data=obj.initiate_data_injestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
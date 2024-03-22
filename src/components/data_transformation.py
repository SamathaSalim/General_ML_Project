import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


#initialising path to save preproccesd input as a pickle file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transormation_config=DataTransformationConfig()

    def get_data_transformer_object(self):#defining the transformations to be done on data
        try:
            numerical_columns=["writing_score","reading_score"] #define num, and categorical features
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),#handles missing values set them to median
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))

                ]
            )

            logging.info(f"numerical_columns : {numerical_columns}")
            logging.info(f"categorical_columns : {categorical_columns}")   

            

            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),
                 ("cat_pipeline",cat_pipeline,categorical_columns)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(sys,e)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("test and train data reading done")

            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying transormation on tarining and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saved preprocessed object")

            save_object(
                self.data_transormation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transormation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            

            
            

    
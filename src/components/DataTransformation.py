import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.Exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    data_transformation_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def TransformData(self):
        try:
            cat_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_colums = ['reading_score', 'writing_score']

            # Create Category Pipline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder()),
                    ('StandardScler',StandardScaler())
                ]
            )

            # Create Num Pipline
            num_pipeline= Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='mean')),
                    ('StandardScaler',StandardScaler())
                ]
            )
            # Join both Pipelines
            preprocessor= make_column_transformer(
                ('num_pipeline',num_pipeline,num_colums),
                ('cat_pipeline',cat_pipeline,cat_columns)
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
    
    def Initiate_Transform_Data(self,train_path,test_path):
        try:
            # read train and test data
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_column= "math_score"
            preprocessor_obj = self.TransformData()
            input_features_train = train_df.drop(columns=[target_column],axis=1)
            # Set Target Feature
            target_feature_train = train_df[target_column]
            input_features_test = test_df.drop(columns=[target_column],axis=1)
            target_feature_test = test_df[target_column]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessor_obj.transform(input_features_train)

            train_arr = np.c_[input_features_test_arr,np.array (target_feature_train)]

            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test)]

            save_object(
                filepath = self.data_transformation_config.data_transformation_path,
                obj= preprocessor_obj
            )

            return (train_arr,test_arr,self.data_transformation_config.data_transformation_path)
        




        except Exception as e:
            raise CustomException(e,sys) 
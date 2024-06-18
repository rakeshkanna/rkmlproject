import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.Exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class Model_Trainer_Config:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()

    def Train_Model(self,train_array,test_array):
        try:
            logging.info("Split the Train and Test Arrays")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models =   {
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbour":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boosting":GradientBoostingRegressor()
            }

            params ={
                "RandomForest":{
                    'n_estimators':[8,16,32,64,128,264],
                },
                "DecisionTree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    #'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.01,0.05,.001]
                    },
                "Linear Regression":{},
                "K-Neighbour":{
                    'n_neighbors':[5,7,9,11]
                },
                "XGBClassifier":{
                    'learning_rate':[0.1,0.01,0.05,.001],
                    'n_estimators':[8,16,32,64,128,264]
                },
                "CatBoosting Classifier":{
                    'depth':[6,8,10],
                    'learning_rate':[0.1,0.01,0.05,.001],
                    'iterations':[30,50,100]
                },
                "AdaBoost Classifier": {
                    'learning_rate':[0.1,0.01,0.05,.001],
                    'n_estimators':[8,16,32,64,128,264]
                }
            }

            model_report = evaluate_models(X_train,y_train,X_test,y_test,models,params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return (r2_square,best_model_name)

        except Exception as e:
            raise CustomException(e,sys)
import sys
import pandas as pd
from src.Exception import CustomException
from src.utils import load_object

class predictionpipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor= load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



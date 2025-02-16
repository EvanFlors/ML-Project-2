import sys, os

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

class NetworkModel:
  
  def __init__(self, preprocessor, model):
    try:
      self.preprocessor = preprocessor
      self.model = model
    except Exception as e:
      raise CustomException(e, sys)
    
  def predict(self, X):
    try:
      transformed_input = self.preprocessor.transform(X)
      y_hat = self.model.predict(transformed_input)
      return y_hat
    except Exception as e:
      raise CustomException(e, sys)
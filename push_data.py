import os
import sys
import json
import pandas as pd
import numpy as np
import pymongo

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from dotenv import load_dotenv
import certifi

load_dotenv()
MONGO_DB_URL = os.getenv('MONGO_DB_URL')

ca = certifi.where()

class NetworkDataExtract():
  def __init__(self):
    try:
      pass
    except Exception as e:
      raise CustomException(e, sys)
    
  def cv_to_json_converter(self, file_path):
    try:
      data = pd.read_csv(file_path)
      data.reset_index(drop = True, inplace = True)
      records = list(json.loads(data.T.to_json()).values())
      return records
    except Exception as e:
      raise CustomException(e, sys)
    
  def insert_data_mongo(self, records, database, collection):
    try:
      self.database = database 
      self.collection = collection
      self.records = records
      
      self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
      self.database = self.mongo_client[self.database]
      self.collection = self.database[self.collection]

      self.collection.insert_many(self.records)
      return len(self.records)
    except Exception as e:  
      raise CustomException(e, sys)
    
if __name__ == '__main__':
  FILE_PATH = "Network_Data\phisingData.csv"
  DATABASE = "NetworkSecurity"
  COLLECTION = "NetworkData"
  
  NetworkDataExtract = NetworkDataExtract()
  records = NetworkDataExtract.cv_to_json_converter(file_path = FILE_PATH)
  len_records = NetworkDataExtract.insert_data_mongo(records = records, database = DATABASE, collection = COLLECTION)
  print(f"Number of records inserted: {len_records}")
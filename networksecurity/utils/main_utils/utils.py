import yaml
from networksecurity.exception.exception import CustomException
from networksecurity.logging import logger
import os, sys
import numpy as np
import dill
import pickle

def read_yaml_file(file_path: str) -> dict:
  try:
    with open(file_path, 'rb') as yaml_file:
      config = yaml.safe_load(yaml_file)
    return config
  except Exception as e:
    raise CustomException(e, sys)
  
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
  try:
    if replace:
      if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as yaml_file:
      yaml.dump(content, yaml_file)
  except Exception as e:
    raise CustomException(e, sys)
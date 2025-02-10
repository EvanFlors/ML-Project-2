import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import CustomException
from networksecurity.logging import logger
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

if __name__ == "__main__":
  try:
    trainingPipelineConfig = TrainingPipelineConfig()
    dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
    data_ingestion = DataIngestion(dataIngestionConfig)
  
    logger.logging.info("Entered main function")
    
    dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
    print(dataIngestionArtifact)
    
  except Exception as e:
    raise CustomException(e, sys)
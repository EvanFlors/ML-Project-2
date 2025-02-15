import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import CustomException
from networksecurity.logging import logger
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

if __name__ == "__main__":
  try:
    trainingPipelineConfig = TrainingPipelineConfig()
    dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
    data_ingestion = DataIngestion(dataIngestionConfig)
  
    logger.logging.info("Initiate the data ingestion.")
    dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
    logger.logging.info("Data ingestion completed.")
    print(dataIngestionArtifact)

    dataValidationConfig = DataValidationConfig(trainingPipelineConfig)
    data_validation = DataValidation(dataValidationConfig, dataIngestionArtifact)

    logger.logging.info("Initiate the data validation.")
    dataValidationArtifact =data_validation.initiate_data_validation()
    logger.logging.info("Data validation completed.")
    print(dataValidationArtifact)
    
  except Exception as e:
    raise CustomException(e, sys)
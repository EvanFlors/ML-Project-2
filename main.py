import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

from networksecurity.exception.exception import CustomException
from networksecurity.logging import logger

if __name__ == "__main__":
  try:
    trainingPipelineConfig = TrainingPipelineConfig()
  
    logger.logging.info("Initiate the data ingestion.")
    dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
    data_ingestion = DataIngestion(dataIngestionConfig)
    dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
    print(dataIngestionArtifact)
    logger.logging.info("Data ingestion completed.")

    logger.logging.info("Initiate the data validation.")
    dataValidationConfig = DataValidationConfig(trainingPipelineConfig)
    data_validation = DataValidation(dataValidationConfig, dataIngestionArtifact)
    dataValidationArtifact = data_validation.initiate_data_validation()
    print(dataValidationArtifact)
    logger.logging.info("Data validation completed.")
    
    logger.logging.info("Initiate the data transformation.")
    dataTransformationConfig = DataTransformationConfig(trainingPipelineConfig)
    data_transformation = DataTransformation(dataTransformationConfig, dataValidationArtifact)
    dataTransformationArtifact = data_transformation.initiate_data_transformation()
    print(dataValidationArtifact)
    logger.logging.info("Data transformation completed.")
    
    logger.logging.info("Model Training sstared")
    model_trainer_config=ModelTrainerConfig(trainingPipelineConfig)
    model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=dataTransformationArtifact)
    model_trainer_artifact=model_trainer.initiate_model_trainer()
    logger.logging.info("Model Training completed.")
    
  except Exception as e:
    raise CustomException(e, sys)
import sys, os

import logging.logger
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
  DataIngestionConfig,
  DataValidationConfig,
  DataTransformationConfig,
  ModelTrainerConfig,
  TrainingPipelineConfig
)

from networksecurity.entity.artifact_entity import (
  DataIngestionArtifact,
  DataValidationArtifact,
  DataTransformationArtifact,
  ModelTrainerArtifact
)

class TrainingPipeline:
  
  def __init__(self):
    self.training_pipeline_config = TrainingPipelineConfig()
  
  def start_data_ingestion(self) -> DataIngestionArtifact:
    try:
      self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
      logging.info("Initiate the data ingestion.")
      data_ingestion = DataIngestion(self.data_ingestion_config)
      data_ingestion = data_ingestion.initiate_data_ingestion()
      logging.info("Data ingestion completed.")
      return data_ingestion
    except Exception as e:
      raise CustomException(e, sys)
    
  def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
    try:
      data_validation_config = DataValidationConfig(self.training_pipeline_config)
      data_validation = DataValidation(data_validation_config = data_validation_config, data_ingestion_artifact = data_ingestion_artifact)
      logging.info("Initiate the data validation.")
      data_validation_artifact = data_validation.initiate_data_validation()
      logging.info("Data validation completed.")
      return data_validation_artifact
    except Exception as e:
      raise CustomException(e, sys)
    
  def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
    try:
      data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
      data_transformation = DataTransformation(data_transformation_config = data_transformation_config, data_validation_artifact = data_validation_artifact)
      logging.info("Initiate the data transformation.")
      data_transformation_artifact = data_transformation.initiate_data_transformation()
      logging.info("Data transformation completed.")
      return data_transformation_artifact
    except Exception as e:
      raise CustomException(e, sys)
    
  def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
    try:
      model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(self.training_pipeline_config)
      model_trainer = ModelTrainer(model_trainer_config = model_trainer_config, data_transformation_artifact = data_transformation_artifact)
      logging.info("Initiate the model trainer.")
      model_trainer_artifact = model_trainer.initiate_model_trainer()
      logging.info("Model trainer completed.")
      return model_trainer_artifact
    except Exception as e:
      raise CustomException(e, sys)
    
  def run_pipeline(self):
    try:
      data_ingestion_artifact = self.start_data_ingestion()
      data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
      data_transformation_artifact = self.start_data_transformation(data_validation_artifact = data_validation_artifact)
      model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)
      return model_trainer_artifact
    except Exception as e:
      raise CustomException(e, sys)
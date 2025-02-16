import sys, os

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

class ModelTrainer:
  
  def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
    try:
      self.model_trainer_config = model_trainer_config
      self.data_transformation_artifact = data_transformation_artifact
    except Exception as e:
      raise CustomException(e, sys)
    
  def train_model(self, x_train, y_train, x_test, y_test):
    models = {
      "Logistic Regression": LogisticRegression(),
      "K-Nearest Neighbours": KNeighborsClassifier(),
      "Decision Tree": DecisionTreeClassifier(),
      "Random Forest": RandomForestClassifier(verbose = 1),
      "AdaBoost": AdaBoostClassifier(verbose = 1),
      "Gradient Boosting": GradientBoostingClassifier(verbose = 1)
    }
    
    params = {
      "Decision Tree": {
        "criterion": ["gini", "entropy"],
        #"splitter": ["best", "random"],
        #"max_features": [None, "sqrt", "log2"],
      },
      "Random Forest": {
        "criterion": ["gini", "entropy", "log_loss"],
        #"max_features": [None, "sqrt", "log2"],
        #"n_estimators": [8, 16, 32, 64, 128, 256],
      },
      "AdaBoost": {
        "n_estimators": [8, 16, 32, 64, 128, 256],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
      },
      "Gradient Boosting": {
        #"n_estimators": [8, 16, 32, 64, 128, 256],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        #"max_features": [None, "sqrt", "log2"],
        #"loss": ["exponential", "log_loss"],
        "subsample": [0.1, 0.2, 0.3, 0.4, 0.5],
        #"criterion": ["friedman_mse", "squared_error"],
      },
      "K-Nearest Neighbours": {
        "n_neighbors": [5, 10, 15, 20, 25],
        #"weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
      },
      "Logistic Regression": {
        "C": [0.1, 0.2, 0.3, 0.4, 0.5],
        #"penalty": ["l1", "l2"],
        #"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
      }
    }
    
    model_report: dict = evaluate_models(
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      y_test = y_test,
      models = models,
      params = params
    )
    
    best_model_score = max(sorted(model_report.values()))
    
    best_model_name = list(model_report.keys())[
      list(model_report.values()).index(best_model_score)
    ]
    
    best_model = models[best_model_name]
    y_train_prediction = best_model.predict(x_train)
    
    classification_train_metric = get_classification_metric(y_true = y_train, y_pred = y_train_prediction)
    
    y_test_pred = best_model.predict(x_test)
    classification_test_metric = get_classification_metric(y_true = y_test, y_pred = y_test_pred)
    
    preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
    
    model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
    os.makedirs(model_dir_path, exist_ok=True)
    
    network_model = NetworkModel(
      model = best_model,
      preprocessor = preprocessor
    )
    
    save_object(
      file_path = self.model_trainer_config.trained_model_file_path,
      obj = network_model
    )
    
    model_trainer_artifact = ModelTrainerArtifact(
      trained_model_file_path = self.model_trainer_config.trained_model_file_path,
      test_metric_artifact = classification_test_metric,
      train_metric_artifact = classification_train_metric
    )
    
    logging.info(f"Model trainer artifact: {model_trainer_artifact}")
    
    return model_trainer_artifact
    
  def initiate_model_trainer(self) -> ModelTrainerArtifact:
    try:
      train_file_path = self.data_transformation_artifact.transformed_train_file_path
      test_file_path = self.data_transformation_artifact.transformed_test_file_path
      
      train_arr = load_numpy_array_data(train_file_path)
      test_arr = load_numpy_array_data(test_file_path)
      
      x_train, y_train, x_test, y_test = (
        train_arr[:, :-1],
        train_arr[:, -1],
        test_arr[:, :-1],
        test_arr[:, -1],
      )
      
      model_trainer_artifact = self.train_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
      
      return model_trainer_artifact
      
    except Exception as e:
      raise CustomException(e, sys)
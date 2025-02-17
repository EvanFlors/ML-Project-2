import sys
import os
import certifi
from dotenv import load_dotenv
import pymongo

from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.exception.exception import CustomException
from networksecurity.utils.main_utils.utils import load_object

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
import pandas as pd

load_dotenv()
ca = certifi.where()
MONGO_DB_URL = os.getenv('MONGO_DB_URL')

client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile = ca)
db = client[DATA_INGESTION_DATABASE_NAME]
collection = db[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

templates = Jinja2Templates(directory = "templates")

app.add_middleware(
  CORSMiddleware,
  allow_origins = origins,
  allow_credentials = True,
  allow_methods = ["*"],
  allow_headers = ["*"],
)

@app.get("/", tags = ["authentication"])
async def index():
  return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
  try:
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()
    return Response("Training is successful")
  except Exception as e:
    raise CustomException(e, sys)
  
@app.get("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
  try:
    df = pd.read_csv(file.file)
    preprocessor = load_object(file_path = "artifacts/preprocessor.pkl")
    final_model = load_object(file_path = "artifacts/model.pkl")
    network_model = NetworkModel(preprocessor = preprocessor, model = final_model)
    print(df.iloc[0])
    y_pred = network_model.predict(df)
    print(y_pred)
    df['predicted_column'] = y_pred
    print(df["predicted_column"])
    df.to_csv("prediction_output/output.csv", index = False)
    table_html = df.to_html(classes = "table table-striped")
    return templates.TemplateResponse("predict.html", {"request": request, "table": table_html})
  except Exception as e:
    raise CustomException(e, sys)

if __name__ == "__main__":
  app_run(app, host = "localhost", port = 8000)
  print("App is running http://localhost:8000")
# Put the code for your API here.
from fastapi import FastAPI
from lightgbm import Booster
from starter.ml.data import process_data
from starter.ml.model import inference
import pickle
from pydantic import BaseModel, Field
import pandas as pd
import os

root_dir = os.path.dirname(os.path.realpath('__file__'))

class PayloadItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    educationnum: int = Field(alias="education-num")
    maritalstatus: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capitalgain: int = Field(alias="capital-gain")
    capitalloss: int = Field(alias="capital-loss")
    hoursperweek: int = Field(alias="hours-per-week")
    nativecountry: str = Field(alias="native-country")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 21740,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }
    }

app = FastAPI()

# Loading starting up the app to avoid latency
@app.on_event("startup")
async def startup_event(): 
    global MODEL, ENCODER, LB
    # Building path to be imported
    encoder_path = os.path.join(root_dir, 'model', 'encoder') if 'starter' in root_dir else os.path.join(root_dir, 'starter', 'model', 'encoder')
    lb_path = os.path.join(root_dir, 'model', 'lb') if 'starter' in root_dir else os.path.join(root_dir, 'starter', 'model', 'lb')
    model_path = os.path.join(root_dir, 'model', 'model.txt') if 'starter' in root_dir else os.path.join(root_dir, 'starter', 'model', 'model.txt')

    if os.path.isfile(encoder_path):
        ENCODER = pickle.load(open(encoder_path, "rb"))
    if os.path.isfile(lb_path):
        LB = pickle.load(open(lb_path, "rb"))
    if os.path.isfile(model_path):
        MODEL = Booster(model_file=model_path)

@app.get("/")
async def say_hello():
    return {"greeting": "Hello Mate Gabriel!"}

@app.post("/prediction/")
async def output(payload: PayloadItem):

    # Building path to be imported
    encoder_path = os.path.join(root_dir, 'model', 'encoder') if 'starter' in root_dir else os.path.join(root_dir, 'starter', 'model', 'encoder')
    lb_path = os.path.join(root_dir, 'model', 'lb') if 'starter' in root_dir else os.path.join(root_dir, 'starter', 'model', 'lb')
    model_path = os.path.join(root_dir, 'model', 'model.txt') if 'starter' in root_dir else os.path.join(root_dir, 'starter', 'model', 'model.txt')

    json_file = {
        "age": payload.age,
        "workclass": payload.workclass,
        "fnlgt": payload.fnlgt,
        "education": payload.education,
        "education-num": payload.educationnum,
        "marital-status": payload.maritalstatus,
        "occupation": payload.occupation,
        "relationship": payload.relationship,
        "race": payload.race,
        "sex": payload.sex,
        "capital-gain": payload.capitalgain,
        "capital-loss": payload.capitalloss,
        "hours-per-week": payload.hoursperweek,
        "native-country": payload.nativecountry
    }

    # Creating DataFrame to run the model with
    data = pd.DataFrame([json_file])
    
    # Importing models
    if os.path.isfile(encoder_path):
        ENCODER = pickle.load(open(encoder_path, "rb"))
    if os.path.isfile(lb_path):
        LB = pickle.load( open(lb_path, "rb"))
    if os.path.isfile(model_path):
        MODEL = Booster(model_file=model_path)
    
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Preprocessing with the encoder
    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label=None, training=False,
        encoder=ENCODER, lb=LB
    )

    # Building the inference model
    prediction = inference(MODEL, X)

    return {"response": int(prediction)}
    
    
    



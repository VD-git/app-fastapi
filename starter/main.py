# Put the code for your API here.
from fastapi import FastAPI
from lightgbm import Booster
from starter.ml.data import process_data
from starter.ml.model import inference
import pickle
from pydantic import BaseModel
import pandas as pd
import os

class PayloadItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    educationnum: int
    maritalstatus: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capitalgain: int
    capitalloss: int
    hoursperweek: int
    nativecountry: str

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/prediction/")
async def output(payload: PayloadItem):

    # Building path to be imported
    encoder_path = os.path.join(os.getcwd(), 'model', 'encoder')
    lb_path = os.path.join(os.getcwd(), 'model', 'lb')
    model_path = os.path.join(os.getcwd(), 'model', 'model.txt')
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
    with open(encoder_path, "rb") as f: 
        encoder = pickle.load(f)
    with open(lb_path, "rb") as f: 
        lb = pickle.load(f)
    model = Booster(model_file=model_path)
    
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Preprocessing with the encoder
    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label=None, training=False,
        encoder=encoder, lb=lb
    )

    # Building the inference model
    prediction = inference(model, X)

    return {"response": int(prediction)}
    
    
    



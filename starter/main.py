# Put the code for your API here.
from fastapi import FastAPI
from typing import Union 
from pydantic import BaseModel
import pandas as pd

class PayloadItem(BaseModel):
    age: int 
    workclass: str
    fnlgt: int 
    education: str
    education-num: int 
    marital-status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital-gain: int 
    capital-loss: int 
    hours-per-week: int 
    native-country: str

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/prediction/")
async def output(payload: PayloadItem):
    data = pd.DataFrame(payload)
    with open(encoder_path, "rb") as f: 
        encoder = pickle.load(f)
    with open(lb_path, "rb") as f: 
        lb = pickle.load(f)
    model = Booster(model_file=model_path)

    y_pred = inference(model, X_test)
    
    
    



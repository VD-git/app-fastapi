from lightgbm import Booster
import json
import pandas as pd
import pickle
import pytest
from starter.ml.data import (process_data, remove_spaces)
from starter.ml.model import (inference)

def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--json", action="store")
    parser.addoption("--encoder", action="store")
    parser.addoption("--lb", action="store")
    parser.addoption("--model", action="store")


@pytest.fixture(scope='session')
def data(request):
    raw_data = pd.read_csv(request.config.option.csv)
    df = remove_spaces(raw_data)
    return df


@pytest.fixture(scope='session')
def metrics(request):
    with open(request.config.option.json) as f:
        mt = json.load(f)
    return mt

@pytest.fixture(scope='session')
def real_predictions(request):
    with open(request.config.option.encoder, "rb") as f: 
        encoder = pickle.load(f)
    with open(request.config.option.lb, "rb") as f: 
        lb = pickle.load(f)
    model = Booster(model_file=request.config.option.model)

    cat_features = ["workclass", "education", "marital-status",\
                    "occupation", "relationship", "race", "sex", "native-country"]

    raw_data = pd.read_csv(request.config.option.csv)
    df = remove_spaces(raw_data)
    
    X, y, encoder, lb = process_data(
        df, categorical_features=cat_features, label="salary", training=False,\
        encoder=encoder, lb=lb
    )

    y_pred = inference(model, X)
    
    return y, y_pred





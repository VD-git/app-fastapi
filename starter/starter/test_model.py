import json
import pandas as pd
import requests
from starter.ml.model import (compute_model_metrics)
import os
from fastapi.testclient import TestClient
from main import app

def test_number_of_rows(data):
    assert len(data) > 1000, f"The amount of chunk tested should be greater than 1000, not {len(data)}"

def test_cols_names(data):
    expected_cols = ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status',\
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',\
                     'hours-per-week', 'native-country', 'salary']
    assert (data.columns == expected_cols).all(), f"Unexpected columns were introduced in the data set:\nAchieved:\n{data.columns}\nExpected\n{expected_cols}"

def test_object_cols(data):
    object_type = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']
    assert (data.dtypes[data.dtypes == 'object'].index == object_type).all(), f"Unexpected object columns:\nAchieved:\n{data.dtypes[data.dtypes == 'object'].index}\nExpected\n{object_type}"

def test_integer_cols(data):
    integer_type = ['age', 'fnlgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    assert (data.dtypes[data.dtypes == 'int64'].index == integer_type).all(), f"Unexpected integer columns:\nAchieved:\n{data.dtypes[data.dtypes == 'int64'].index}\nExpected\n{integer_type}"

def test_output_type(real_predictions):
    y_real, y_pred = real_predictions
    assert y_pred.dtype == 'int64', f"Check the output type, it was found a type of {y_pred.dtype}, while it was expected int64"

def test_metric_precision(metrics, real_predictions):
    y_real, y_pred = real_predictions
    precision, recall, fbeta = compute_model_metrics(y_real, y_pred)
    sample_metric = metrics.get("precision")
    assert metrics.get("precision") < precision, f"Precision for the whole data set is expected to be greater than only for test: whole data set - {precision} vs. test - {sample_metric}"

def test_metric_recall(metrics, real_predictions):
    y_real, y_pred = real_predictions
    precision, recall, fbeta = compute_model_metrics(y_real, y_pred)
    sample_metric = metrics.get("recall")
    assert metrics.get("recall") < recall, f"Recall for the whole data set is expected to be greater than only for test: whole data set - {recall} vs. test - {sample_metric}"

def test_metric_fbeta(metrics, real_predictions):
    y_real, y_pred = real_predictions
    precision, recall, fbeta = compute_model_metrics(y_real, y_pred)
    sample_metric = metrics.get("fbeta")
    assert metrics.get("fbeta") < fbeta, f"Fbeta for the whole data set is expected to be greater than only for test: whole data set - {fbeta} vs. test - {sample_metric}"

def test_api_get_status(server):
    r = requests.get("http://127.0.0.1:8000/")
    assert r.status_code == 200

def test_api_get_output_keys(server):
    r = requests.get("http://127.0.0.1:8000/")
    payload = r.json() #json.load(r)
    assert list(payload.keys())[0] == "greeting"

def test_api_get_output_values(server):
    r = requests.get("http://127.0.0.1:8000/")
    payload = r.json() #json.load(r)
    assert list(payload.values())[0] == "Hello Mate Gabriel!"

def test_api_post_negative_status(server):
    client = TestClient(server)
    payload = {
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
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = requests.post("http://127.0.0.1:8000/prediction/", data = json.dumps(payload))
    assert r.status_code == 200

def test_api_post_negative_content(server):
    payload = {
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
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = requests.post("http://127.0.0.1:8000/prediction/", data = json.dumps(payload))
    output = r.json() #json.load(r)
    assert output['response'] == 0

def test_api_post_positive_status(server):
    payload = {
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
    r = requests.post("http://127.0.0.1:8000/prediction/", data = json.dumps(payload))
    assert r.status_code == 200

def test_api_post_positive_content(server):
    payload = {
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
    r = requests.post("http://127.0.0.1:8000/prediction/", data = json.dumps(payload))
    output = r.json() #json.load(r)
    assert output['response'] == 1

    




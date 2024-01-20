# Import our app from main.py.
from fastapi.testclient import TestClient
import json
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_get_status():
    r = client.get("/")
    assert r.status_code == 200

def test_api_get_output_keys():
    r = client.get("/")
    payload = r.json() #json.load(r)
    assert list(payload.keys())[0] == "greeting"

def test_api_get_output_values():
    r = client.get("/")
    payload = r.json() #json.load(r)
    assert list(payload.values())[0] == "Hello Mate Gabriel!"

def test_api_post_negative_status():
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
    r = client.post("/prediction/", data = json.dumps(payload))
    assert r.status_code == 200

def test_api_post_negative_content():
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
    r = client.post("/prediction/", data = json.dumps(payload))
    output = r.json() #json.load(r)
    assert output['response'] == 0

def test_api_post_positive_status():
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
    r = client.post("/prediction/", data = json.dumps(payload))
    assert r.status_code == 200

def test_api_post_positive_content():
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
    r = client.post("/prediction/", data = json.dumps(payload))
    output = r.json() #json.load(r)
    assert output['response'] == 1
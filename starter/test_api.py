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
    payload = json.load(r)
    assert list(payload.keys())[0] == "greeting"

def test_api_get_output_values():
    r = client.get("/")
    payload = json.load(r)
    assert list(payload.values())[0] == "Hello World!"

def test_api_post_negative():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "educationnum": 13,
        "maritalstatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capitalgain": 2174,
        "capitalloss": 0,
        "hoursperweek": 40,
        "nativecountry": "United-States"
    }
    r = client.post("/prediction/", data = json.dumps(payload))
    output = json.load(r)
    assert output.get('response') == 0

def test_api_post_positive():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "educationnum": 13,
        "maritalstatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capitalgain": 21740,
        "capitalloss": 0,
        "hoursperweek": 40,
        "nativecountry": "United-States"
    }
    r = client.post("/prediction/", data = json.dumps(payload))
    output = json.load(r)
    assert output.get('response') == 1
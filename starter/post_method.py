#!/usr/bin/env python
import argparse
import requests
import json

def go(args):
    r = requests.post(args.url, args.payload)
    status, response = r.status_code, r.json()
    return {"status": status, "response": response.get("response"), "payload": args.payload}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing POST Method API"
    )

    parser.add_argument(
        "--url",
        type=str,
        help="Render URL used for the API",
        required=False,
        default="https://fastapi-application-gl4h.onrender.com/prediction"
    )

    parser.add_argument(
        "--payload",
        type=str,
        help="Payload for the model to make the inference",
        required=False,
        default='{"age": 39, "workclass": "State-gov", "fnlgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 21740, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}'
    )

    args = parser.parse_args()

    print(go(args))

#!/usr/bin/env python

# Script to train machine learning model.
import argparse
import json
from lightgbm import Booster
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import (process_data, remove_spaces)
from ml.model import (compute_model_metrics, inference, model_slices_metrics, train_model)

# Add code to load in the data.
def reading_processing_data(csv:str):
    dataset_path = os.path.join(os.getcwd(), 'data', csv)
    raw_data = pd.read_csv(dataset_path)
    data = remove_spaces(raw_data)
    return data
    
# Optional enhancement, use K-fold cross validation instead of a train-test split.
def preprocessing_data(data, test_size:float):
    train, test = train_test_split(data, test_size=test_size)
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    return train, test, X_train, X_test, y_train, y_test, encoder, lb
    
# Train and save a model.
def training_save_model(X_train, y_train, encoder, lb, params):
    params = json.loads(params)
    LGBM = train_model(X_train, y_train, params=params)

    encoder_path = os.path.join(os.getcwd(), 'model', 'encoder')
    lb_path = os.path.join(os.getcwd(), 'model', 'lb')
    model_path = os.path.join(os.getcwd(), 'model', 'model.txt')

    with open(encoder_path, "wb") as f: 
        pickle.dump(encoder, f)
    
    with open(lb_path, "wb") as f: 
        pickle.dump(lb, f)
    
    LGBM.booster_.save_model(model_path)

def perform_analysis(test, X_test, y_test, cat_features):

    encoder_path = os.path.join(os.getcwd(), 'model', 'encoder')
    lb_path = os.path.join(os.getcwd(), 'model', 'lb')
    model_path = os.path.join(os.getcwd(), 'model', 'model.txt')
    metrics_path = os.path.join(os.getcwd(), 'model', 'metrics.json')

    with open(encoder_path, "rb") as f: 
        encoder = pickle.load(f)
    with open(lb_path, "rb") as f: 
        lb = pickle.load(f)
    model = Booster(model_file=model_path)
    
    # Perform analysis of the metrics of the model, dump a json for future analysis
    y_pred = inference(model, X_test)
    
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    with open(metrics_path, 'w') as f:
        json.dump({'precision': precision, 'recall': recall, 'fbeta': fbeta}, f)

    # Generates a table with the evaluation of the slice data set
    model_slices_metrics(test, y_test, y_pred, list(cat_features)).to_csv(os.path.join(os.getcwd(),'model','slice_output.txt'), index = False)

def go(args):
    os.chdir('/home/vdias94/app-fastapi/starter')
    data = reading_processing_data(csv = args.csv)
    train, test, X_train, X_test, y_train, y_test, encoder, lb = preprocessing_data(data = data, test_size = args.test_size)
    training_save_model(X_train, y_train, encoder, lb, params=args.params)
    perform_analysis(test, X_test, y_test, cat_features=args.cat_features.split(","))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script For Model Training"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="File with the data",
        required=False,
        default="census.csv"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Size of the test split",
        required=False,
        default=0.2
    )

    parser.add_argument(
        "--params",
        type=str,
        help="Parameters to perform the training",
        required=False,
        default='{"num_leaves": [20, 40, 80], "max_depth": [4, 6, 8], "n_estimators": [200, 400]}'
    )

    parser.add_argument(
        "--cat_features",
        type=str,
        help="Categorical features to split the analysis",
        required=False,
        default='workclass,education,marital-status,occupation,relationship,race,sex,native-country'
    )

    args = parser.parse_args()

    go(args)

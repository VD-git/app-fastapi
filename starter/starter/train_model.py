# Script to train machine learning model.
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
os.chdir('/home/vdias94/app-fastapi/starter')
dataset_path = os.path.join(os.getcwd(), 'data', 'census.csv')
raw_data = pd.read_csv(dataset_path)
data = remove_spaces(raw_data)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Train and save a model.
params = {
    'num_leaves': [20, 40, 80],\
    'max_depth': [4, 6, 8],\
    'n_estimators': [200, 400]
}
LGBM = train_model(X_train, y_train, params=params)

encoder_path = os.path.join(os.getcwd(), 'model', 'encoder')
lb_path = os.path.join(os.getcwd(), 'model', 'lb')
model_path = os.path.join(os.getcwd(), 'model', 'model.txt')
metrics_path = os.path.join(os.getcwd(), 'model', 'metrics.json')

with open(encoder_path, "wb") as f: 
    pickle.dump(encoder, f)

with open(lb_path, "wb") as f: 
    pickle.dump(lb, f)

LGBM.booster_.save_model(model_path)

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
model_slices_metrics(test, y_test, y_pred, cat_features).to_csv(os.path.join(os.getcwd(),'model','slice_output.txt'), index = False)

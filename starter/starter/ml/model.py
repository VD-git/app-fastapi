import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, params=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    params : dict
        Dictionary of parameters in order to perform the hyperparameter tuning.
    Returns
    -------
    model
        Trained machine learning model.
    """
    if params is None:
        LGBM=LGBMClassifier(verbose=-1)
        LGBM.fit(X_train, y_train)
    else:
        GS = GridSearchCV(estimator=LGBMClassifier(verbose=-1), param_grid=params)
        GS.fit(X_train, y_train)

        LGBM=LGBMClassifier(**GS.best_params_, verbose=-1)
        LGBM.fit(X_train, y_train)
    return LGBM


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Light Gradient Boosting Method
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = np.array([1 if y >= 0.50 else 0 for y in model.predict(X)])
    return preds

def model_slices_metrics(test: pd.DataFrame, real: np.array, prediction: np.array, slice_cols: list):
    """ Perform a slice analysis of the predictions
    
    Inputs
    ------
    test : pd.DataFrame
        Data set of the test data.
    real : np.array
        Data from the real test output.
    prediction : np.array
        Data from the test predictions.
    slice_cols : list
        List of columns to be sliced
    Returns
    -------
    data : pd.DataFrame
        A table with the sliced performance.
    """
    data = []
    combined_df = pd.concat(\
        [test.reset_index().drop(['index'], axis = 1),\
         pd.DataFrame(real, columns = ['real']),\
         pd.DataFrame(prediction, columns = ['prediction'])\
        ], axis = 1)
    for col in slice_cols:
        for feature in combined_df[col].unique():
            eval_set = combined_df[combined_df[col] == feature]
            precision, recall, fbeta = compute_model_metrics(eval_set['real'], eval_set['prediction'])
            data.append({"column": col, "slice": feature, "size": len(eval_set), "precision": precision, "recall": recall, "fbeta": fbeta})
    return pd.DataFrame(data)

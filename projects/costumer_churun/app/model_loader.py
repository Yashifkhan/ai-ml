# import joblib

# def load_artifacts():
#     log_model = joblib.load("models/logistic_regression_model.pkl")
#     rf_model = joblib.load("models/random_forest_model.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     feature_columns = joblib.load("models/feature_columns.pkl")
#     thresholds = joblib.load("models/thresholds.pkl")
#     return log_model, rf_model, scaler, feature_columns, thresholds


import joblib
import os

# app folder ke relative models folder ka absolute path banate hain
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../project/app
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")       # .../project/models


def load_artifacts():
    log_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    thresholds = joblib.load(os.path.join(MODELS_DIR, "thresholds.pkl"))
    return log_model, rf_model, scaler, feature_columns, thresholds
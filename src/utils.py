import os
import joblib


def save_model(model, path="models/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    return joblib.load(path)

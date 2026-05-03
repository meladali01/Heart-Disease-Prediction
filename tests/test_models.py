import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.preprocess import build_preprocessor, split_features_target
from src.models import knn_pipeline, tree_pipeline


def make_sample_df():
    # build a tiny dataset with numeric and categorical columns expected by preprocessor
    data = {
        "age": [30, 40, 50, 60],
        "sex": [1, 0, 1, 0],
        "cp": [1, 2, 3, 1],
        "trestbps": [120, 130, 140, 150],
        "chol": [200, 210, 220, 230],
        "fbs": [0, 0, 1, 0],
        "restecg": [0, 1, 0, 1],
        "thalach": [150, 140, 130, 120],
        "exang": [0, 1, 0, 1],
        "oldpeak": [1.0, 2.0, 1.5, 0.5],
        "slope": [2, 2, 1, 1],
        "ca": [0, 1, 0, 1],
        "thal": [3, 2, 3, 2],
        "target": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_knn_pipeline_fit_predict():
    df = make_sample_df()
    X, y = split_features_target(df)
    pre = build_preprocessor(df)
    pipe = knn_pipeline(pre, n_neighbors=1)
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    # predictions should be 0/1
    assert set(preds.tolist()).issubset({0, 1})


def test_tree_pipeline_fit_predict():
    df = make_sample_df()
    X, y = split_features_target(df)
    pre = build_preprocessor(df)
    pipe = tree_pipeline(pre, max_depth=3, random_state=0)
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds.tolist()).issubset({0, 1})

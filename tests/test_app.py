import os
import importlib
import joblib
import pandas as pd
from src.preprocess import build_preprocessor, split_features_target
from src.models import knn_pipeline


def make_sample_df():
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


def test_app_predict_endpoint(monkeypatch, tmp_path):
    # prepare sample data and a trained model saved to a temp file
    df = make_sample_df()
    X, y = split_features_target(df)
    pre = build_preprocessor(df)
    pipe = knn_pipeline(pre, n_neighbors=1)
    pipe.fit(X, y)

    model_file = tmp_path / "temp_model.joblib"
    joblib.dump(pipe, str(model_file))

    # monkeypatch data.load_data so app.ModelLoader discovers correct feature columns
    import src.data as data_mod

    monkeypatch.setattr(data_mod, "load_data", lambda path="data/heart.csv": df)

    # set MODEL_PATH env before importing app so the loader uses our temp model
    os.environ["MODEL_PATH"] = str(model_file)

    # import (or reload) app to pick up env and monkeypatched load_data
    import src.app as app_mod
    importlib.reload(app_mod)

    client = app_mod.app.test_client()

    # health check
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json().get("status") == "ok"

    # build a single input from df first row
    feature_cols = df.drop(columns=["target"]).columns.tolist()
    row = df.iloc[0].drop(labels=["target"]).to_dict()

    # send as dict
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 200
    j = r.get_json()
    assert "prediction" in j
    assert "probability" in j

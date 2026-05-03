import os
import tempfile
import pandas as pd
from src.data import load_data


def test_load_data_reads_csv(tmp_path):
    # create a small dataframe with 'target' column
    df = pd.DataFrame({
        "age": [50, 60],
        "sex": [1, 0],
        "cp": [1, 2],
        "trestbps": [120, 140],
        "chol": [230, 250],
        "fbs": [0, 1],
        "restecg": [1, 0],
        "thalach": [150, 130],
        "exang": [0, 1],
        "oldpeak": [2.3, 1.4],
        "slope": [2, 1],
        "ca": [0, 1],
        "thal": [3, 2],
        "target": [0, 1],
    })
    p = tmp_path / "test_heart.csv"
    df.to_csv(p, index=False)

    loaded = load_data(str(p))
    assert isinstance(loaded, pd.DataFrame)
    # columns preserved
    assert list(df.columns) == list(loaded.columns)
    # values preserved
    assert loaded.shape == (2, len(df.columns))

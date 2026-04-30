import os
import pandas as pd
import requests


def download_uc_heart(dest_path="data/heart.csv"):
    """Download the UCI processed Cleveland heart disease dataset and save as CSV."""
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    )
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    r = requests.get(url)
    r.raise_for_status()
    text = r.text.strip().splitlines()
    cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    rows = [line.split(",") for line in text if line.strip()]
    df = pd.DataFrame(rows, columns=cols)
    df.replace("?", pd.NA, inplace=True)
    # convert numeric columns
    for c in [c for c in df.columns if c != "thal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 'thal' may contain missing values; coerce
    df["thal"] = pd.to_numeric(df["thal"], errors="coerce")
    # binary target: 0 => no disease, >0 => disease
    df["target"] = df["num"].fillna(0).apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=["num"], inplace=True)
    df.to_csv(dest_path, index=False)
    return dest_path


def load_data(path="data/heart.csv"):
    if not os.path.exists(path):
        download_uc_heart(path)
    return pd.read_csv(path)

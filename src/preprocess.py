from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd


def build_preprocessor(df):
    # explicit categorical columns for this dataset
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric = [c for c in df.columns if c not in categorical + ["target"]]

    # transformers wrapped as Pipelines
    num_transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_transform = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transform, numeric),
            ("cat", cat_transform, categorical),
        ]
    )
    return preprocessor


def split_features_target(df):
    X = df.drop(columns=["target"]) 
    y = df["target"].astype(int)
    return X, y

from .data import load_data
import joblib
import pandas as pd


def run_local_test(model_path="models/knn_final.joblib"):
    df = load_data()
    feature_cols = [c for c in df.columns if c != "target"]

    # Use the dataset median as a safe sample
    sample = df[feature_cols].median().to_dict()
    X = pd.DataFrame([sample], columns=feature_cols)

    model = joblib.load(model_path)
    pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X).max()

    print("Sample features:")
    print(X.to_dict(orient="records")[0])
    print("Prediction:", int(pred))
    print("Probability:", float(prob) if prob is not None else None)


if __name__ == "__main__":
    run_local_test()

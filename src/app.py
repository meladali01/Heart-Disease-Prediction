from flask import Flask, request, jsonify
import joblib
import pandas as pd
from .data import load_data

app = Flask(__name__)

# Try to load the best-performing model saved earlier
MODEL_PATH = "models/knn_final.joblib"
model = joblib.load(MODEL_PATH)

# Load dataset to get expected feature columns
df = load_data()
FEATURE_COLS = [c for c in df.columns if c != "target"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    # Accept either dict of features or list of values
    if "features" not in data:
        return jsonify({"error": "provide 'features' as dict or list"}), 400

    feats = data["features"]
    if isinstance(feats, list):
        if len(feats) != len(FEATURE_COLS):
            return (
                jsonify({"error": f"expected {len(FEATURE_COLS)} features"}),
                400,
            )
        row = dict(zip(FEATURE_COLS, feats))
    elif isinstance(feats, dict):
        row = {k: feats.get(k, None) for k in FEATURE_COLS}
    else:
        return jsonify({"error": "'features' must be list or dict"}), 400

    X = pd.DataFrame([row], columns=FEATURE_COLS)
    try:
        pred = int(model.predict(X)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X).max())
        return jsonify({"prediction": pred, "probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

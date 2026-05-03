from flask import Flask, request, jsonify, current_app
import threading
import joblib
import pandas as pd
from .data import load_data
import os

app = Flask(__name__)

# Configuration: model path can be overridden via env or app config
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/knn_final.joblib")


class ModelLoader:
    """Thread-safe lazy loader for the model and its expected feature columns."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._feature_cols = None
        self._lock = threading.Lock()

    def load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not os.path.exists(self.model_path):
                        raise FileNotFoundError(f"Model not found at {self.model_path}")
                    self._model = joblib.load(self.model_path)
                    # load a small sample of data to determine feature columns
                    df = load_data()
                    self._feature_cols = [c for c in df.columns if c != "target"]
        return self._model

    @property
    def feature_cols(self):
        if self._feature_cols is None:
            # load will populate feature cols
            self.load()
        return self._feature_cols


# Attach a loader instance to the app for reuse
with app.app_context():
    current_app.model_loader = ModelLoader()


@app.route("/health", methods=["GET"])
def health():
    # quick health check; don't force model load
    return jsonify({"status": "ok"})


def _prepare_input(data, feature_cols):
    if not data:
        raise ValueError("missing JSON body")
    if "features" not in data:
        raise ValueError("provide 'features' as dict or list")

    feats = data["features"]
    if isinstance(feats, list):
        if len(feats) != len(feature_cols):
            raise ValueError(f"expected {len(feature_cols)} features")
        row = dict(zip(feature_cols, feats))
    elif isinstance(feats, dict):
        row = {k: feats.get(k, None) for k in feature_cols}
    else:
        raise ValueError("'features' must be list or dict")

    X = pd.DataFrame([row], columns=feature_cols)
    return X


@app.route("/predict", methods=["POST"])
def predict():
    try:
        loader = current_app.model_loader
        model = loader.load()
        feature_cols = loader.feature_cols
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"failed to load model: {e}"}), 500

    try:
        data = request.get_json()
        X = _prepare_input(data, feature_cols)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    try:
        pred = int(model.predict(X)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X).max())
        return jsonify({"prediction": pred, "probability": prob})
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 500


if __name__ == "__main__":
    # Allow overriding host/port via env vars for flexibility
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", 5000))
    app.run(host=host, port=port, debug=True)

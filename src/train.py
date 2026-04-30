import argparse
import os

from sklearn.model_selection import train_test_split

from .data import load_data, download_uc_heart
from .preprocess import build_preprocessor, split_features_target
from .models import knn_pipeline, tree_pipeline
from .evaluate import evaluate_model
from .utils import save_model


def main():
    parser = argparse.ArgumentParser(description="Train a heart disease classifier")
    parser.add_argument("--model", choices=["knn", "tree"], default="knn")
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--data", type=str, default="data/heart.csv")
    parser.add_argument("--output", type=str, default="models/model.joblib")
    args = parser.parse_args()

    print("Loading data...")
    os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
    df = load_data(args.data)

    X, y = split_features_target(df)
    preprocessor = build_preprocessor(df)

    print(f"Building {args.model} pipeline...")
    if args.model == "knn":
        pipeline = knn_pipeline(preprocessor, n_neighbors=args.n_neighbors)
    else:
        pipeline = tree_pipeline(preprocessor, max_depth=args.max_depth, random_state=args.random_state)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    results = evaluate_model(pipeline, X_test, y_test)
    for k, v in results.items():
        if k == "report":
            print("Classification report:\n", v)
        else:
            print(f"{k}: {v}")

    out = save_model(pipeline, args.output)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()

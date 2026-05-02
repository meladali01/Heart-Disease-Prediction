import argparse
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from .data import load_data
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
    parser.add_argument("--cv", type=int, default=0, help="Run stratified K-fold CV (n_splits). 0 disables CV.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for CV/training")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model file")
    parser.add_argument("--save_metrics", type=str, default=None, help="Path to save evaluation metrics as JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        print("Loading data...")
    df = load_data(args.data)

    # Ensure output path availability
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output file {args.output} exists. Use --overwrite to replace it.")

    X, y = split_features_target(df)
    preprocessor = build_preprocessor(df)

    print(f"Building {args.model} pipeline...")
    if args.model == "knn":
        pipeline = knn_pipeline(preprocessor, n_neighbors=args.n_neighbors)
    else:
        pipeline = tree_pipeline(preprocessor, max_depth=args.max_depth, random_state=args.random_state)

    # Optionally run cross-validation on full dataset
    if args.cv and args.cv > 1:
        if args.verbose:
            print(f"Running {args.cv}-fold cross-validation (n_jobs={args.n_jobs})...")
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=args.n_jobs)
        print(f"CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")

    if args.verbose:
        print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    print("Training...")
    pipeline.fit(X_train, y_train)

    if args.verbose:
        print("Evaluating...")
    results = evaluate_model(pipeline, X_test, y_test)
    for k, v in results.items():
        if k == "report":
            print("Classification report:\n", v)
        else:
            print(f"{k}: {v}")
    out = save_model(pipeline, args.output)
    print(f"Saved model to {out}")

    # Optionally save metrics
    if args.save_metrics:
        metrics = {"evaluation": results}
        if args.cv and args.cv > 1:
            metrics["cv"] = {"n_splits": args.cv, "accuracy_mean": float(scores.mean()), "accuracy_std": float(scores.std()), "scores": [float(s) for s in scores]}
        with open(args.save_metrics, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved metrics to {args.save_metrics}")


if __name__ == "__main__":
    main()

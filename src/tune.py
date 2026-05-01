import joblib
from sklearn.model_selection import GridSearchCV, train_test_split

from .data import load_data
from .preprocess import build_preprocessor, split_features_target
from .models import knn_pipeline, tree_pipeline
from .evaluate import evaluate_model
from .utils import save_model


def tune_tree_and_compare(data_path="data/heart.csv", random_state=42, test_size=0.2):
    df = load_data(data_path)
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # baseline KNN
    knn = knn_pipeline(preprocessor, n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_res = evaluate_model(knn, X_test, y_test)

    # tree with grid search
    tree_pipe = tree_pipeline(preprocessor)
    param_grid = {
        "clf__max_depth": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "clf__min_samples_leaf": [1, 2, 3, 4],
    }
    gs = GridSearchCV(tree_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_tree = gs.best_estimator_
    cv_best = gs.best_score_

    tree_res = evaluate_model(best_tree, X_test, y_test)

    print("\n=== Baseline KNN ===")
    for k, v in knn_res.items():
        if k == "report":
            print("Classification report:\n", v)
        else:
            print(f"{k}: {v}")

    print("\n=== Tuned Decision Tree ===")
    print(f"Best CV f1: {cv_best}")
    print(f"Best params: {gs.best_params_}")
    for k, v in tree_res.items():
        if k == "report":
            print("Classification report:\n", v)
        else:
            print(f"{k}: {v}")

    # save models
    save_model(knn, "models/knn_final.joblib")
    save_model(best_tree, "models/tree_tuned.joblib")
    print("\nSaved models to models/knn_final.joblib and models/tree_tuned.joblib")


if __name__ == "__main__":
    tune_tree_and_compare()

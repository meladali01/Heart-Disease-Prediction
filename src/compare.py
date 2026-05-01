from sklearn.model_selection import train_test_split

from .data import load_data
from .preprocess import build_preprocessor, split_features_target
from .models import knn_pipeline, tree_pipeline
from .evaluate import evaluate_model
from .utils import save_model


def compare_and_save(data_path="data/heart.csv", random_state=42, test_size=0.2):
    df = load_data(data_path)
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    knn = knn_pipeline(preprocessor, n_neighbors=5)
    tree = tree_pipeline(preprocessor, max_depth=None, random_state=random_state)

    knn.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    knn_res = evaluate_model(knn, X_test, y_test)
    tree_res = evaluate_model(tree, X_test, y_test)

    print("\n=== KNN Results ===")
    for k, v in knn_res.items():
        if k == "report":
            print("Classification report:\n", v)
        else:
            print(f"{k}: {v}")

    print("\n=== Decision Tree Results ===")
    for k, v in tree_res.items():
        if k == "report":
            print("Classification report:\n", v)
        else:
            print(f"{k}: {v}")

    # save models
    save_model(knn, "models/knn_compare.joblib")
    save_model(tree, "models/tree.joblib")
    print("\nSaved models to models/knn_compare.joblib and models/tree.joblib")


if __name__ == "__main__":
    compare_and_save()

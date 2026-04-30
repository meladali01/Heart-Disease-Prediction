from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def knn_pipeline(preprocessor, n_neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def tree_pipeline(preprocessor, max_depth=None, random_state=42):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])

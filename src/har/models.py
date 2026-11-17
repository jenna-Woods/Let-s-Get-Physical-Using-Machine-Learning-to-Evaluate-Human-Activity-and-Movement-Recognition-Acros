from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build(name, seed, quick=False):
    if name == "dummy":
        return Pipeline([("model", DummyClassifier(strategy="most_frequent"))]), {}
    if name == "logistic":
        model = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=seed)
        grid = {"model__C": [1.] if quick else [.01, .1, 1., 10.]}
    elif name == "forest":
        model = RandomForestClassifier(n_estimators=50 if quick else 300,
                                       n_jobs=1, random_state=seed)
        grid = {"model__max_depth": [None] if quick else [None, 20],
                "model__min_samples_leaf": [1] if quick else [1, 3],
                "model__max_features": ["sqrt"] if quick else ["sqrt", .5]}
    elif name == "svm":
        model = SVC(kernel="rbf", cache_size=512)
        grid = {"model__C": [1.] if quick else [.1, 1., 10.],
                "model__gamma": ["scale"] if quick else ["scale", .001, .01]}
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("select", SelectKBest(f_classif)),
                     ("scale", "passthrough" if name == "forest" else StandardScaler()),
                     ("model", model)]), grid

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_splits(y, subjects, protocol, folds, seed):
    if protocol == "grouped":
        if len(np.unique(subjects)) < folds:
            raise ValueError("Too few participants for grouped folds")
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(y)), y, subjects))
    elif protocol == "random":
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(y)), y))
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    for train, valid in splits:
        if np.intersect1d(train, valid).size:
            raise ValueError("Row leakage")
        if protocol == "grouped" and set(subjects[train]) & set(subjects[valid]):
            raise ValueError("Participant leakage")
        if set(y[train]) != set(y) or set(y[valid]) != set(y):
            raise ValueError("A fold is missing an activity")
    return splits

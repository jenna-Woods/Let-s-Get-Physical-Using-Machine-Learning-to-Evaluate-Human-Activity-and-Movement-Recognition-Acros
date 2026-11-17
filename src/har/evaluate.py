from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item()), encoding="utf-8")


def evaluate(estimator, X, y, subjects, rows, labels, output, threads=1):
    from threadpoolctl import threadpool_limits
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=threads):
        estimator.predict(X[:min(32, len(X))])
        times = []
        for _ in range(5):
            start = time.perf_counter()
            predicted = estimator.predict(X)
            times.append(time.perf_counter() - start)
    ids = sorted(labels)
    metrics = {"accuracy": accuracy_score(y, predicted),
               "macro_f1": f1_score(y, predicted, labels=ids, average="macro", zero_division=0),
               "prediction_seconds": float(np.median(times)),
               "prediction_us_per_row": float(np.median(times) / len(y) * 1e6),
               "n_test": len(y)}
    write_json(output / "metrics.json", metrics)
    write_json(output / "classification_report.json", classification_report(
        y, predicted, labels=ids, target_names=[labels[i] for i in ids],
        zero_division=0, output_dict=True))
    predictions = pd.DataFrame({"row": rows, "subject": subjects, "true": y,
                                "predicted": predicted})
    predictions.to_csv(output / "predictions.csv", index=False)
    per_subject = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        per_subject.append({"subject": int(subject), "n": int(mask.sum()),
                            "accuracy": accuracy_score(y[mask], predicted[mask]),
                            "macro_f1": f1_score(y[mask], predicted[mask], labels=ids,
                                                 average="macro", zero_division=0)})
    pd.DataFrame(per_subject).to_csv(output / "per_subject.csv", index=False)
    for normalize, suffix in [(None, "counts"), ("true", "normalized")]:
        matrix = confusion_matrix(y, predicted, labels=ids, normalize=normalize)
        pd.DataFrame(matrix, index=[labels[i] for i in ids],
                     columns=[labels[i] for i in ids]).to_csv(output / f"confusion_{suffix}.csv")
    return metrics


def save_features(estimator, features, output):
    if "select" not in estimator.named_steps:
        return
    selector = estimator.named_steps["select"]
    table = features.copy()
    table["score"] = selector.scores_
    table["selected"] = selector.get_support()
    table.to_csv(Path(output) / "features.csv", index=False)

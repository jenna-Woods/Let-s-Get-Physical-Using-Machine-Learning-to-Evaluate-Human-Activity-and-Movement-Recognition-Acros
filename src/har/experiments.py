"""Nested validation; the official test is read only by evaluate_final."""
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from .data import Dataset, audit
from .evaluate import evaluate, save_features, write_json
from .models import build
from .splits import make_splits


def validate_config(config):
    if config["outer_folds"] < 2 or config["inner_folds"] < 2:
        raise ValueError("Need at least two folds")
    if not config["feature_counts"] or any(k not in range(1, 562) for k in config["feature_counts"]):
        raise ValueError("Feature counts must be in 1..561")
    if config["compact_tolerance"] < 0:
        raise ValueError("Tolerance cannot be negative")
    if config["threads"] < 1 or config["search_jobs"] < 1:
        raise ValueError("Thread and worker counts must be positive")
    if not set(config["models"]) <= {"dummy", "logistic", "forest", "svm"}:
        raise ValueError("Unknown model")
    if not config["models"] or not config["protocols"]:
        raise ValueError("Models and protocols cannot be empty")
    if not set(config["protocols"]) <= {"grouped", "random"}:
        raise ValueError("Unknown protocol")


def prepare_output(output, config, dataset):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new output directory; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    write_json(output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "joblib": joblib.__version__,
        "git_revision": revision, "git_dirty": dirty,
        "train_fingerprint": dataset.fingerprint()})
    return output


def search(name, config, X, y, cv):
    estimator, grid = build(name, config["seed"], config["quick"])
    if name != "dummy":
        grid["select__k"] = config["feature_counts"]
    searcher = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv,
                            refit=False, n_jobs=config["search_jobs"], error_score="raise")
    start = time.perf_counter()
    with joblib.parallel_config(backend="loky", inner_max_num_threads=config["threads"],
                                max_nbytes=None):
        with threadpool_limits(limits=config["threads"]):
            searcher.fit(X, y)
    return estimator, pd.DataFrame(searcher.cv_results_), time.perf_counter() - start


def candidates(results, tolerance):
    """Use inner scores only. Return fixed-k winners plus best/compact choices."""
    scores = results.mean_test_score.to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite validation scores")
    params = results.params.tolist()
    ks = sorted({p.get("select__k", 561) for p in params})
    fixed = {}
    for k in ks:
        indices = [i for i, p in enumerate(params) if p.get("select__k", 561) == k]
        fixed[k] = max(indices, key=lambda i: (scores[i], -i))
    best = max(range(len(params)), key=lambda i: (scores[i], -params[i].get("select__k", 561), -i))
    eligible = [i for i in fixed.values() if scores[i] >= scores[best] - tolerance]
    compact = min(eligible, key=lambda i: (params[i].get("select__k", 561), -scores[i]))
    return fixed, best, compact


def fit(estimator, params, X, y, threads):
    model = clone(estimator).set_params(**params)
    start = time.perf_counter()
    with threadpool_limits(limits=threads):
        model.fit(X, y)
    return model, time.perf_counter() - start


def run_cv(data_root, config, output):
    validate_config(config)
    dataset = Dataset(data_root)
    part = dataset.load("train")
    output = prepare_output(output, config, dataset)
    audit(dataset, part, output / "audit")
    metrics = []
    for protocol in config["protocols"]:
        outer = make_splits(part.y, part.subjects, protocol,
                            config["outer_folds"], config["seed"])
        for fold, (train, valid) in enumerate(outer):
            fold_dir = output / protocol / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            write_json(fold_dir / "split.json", {
                "train_rows": train.tolist(), "valid_rows": valid.tolist(),
                "train_subjects": sorted(set(part.subjects[train].tolist())),
                "valid_subjects": sorted(set(part.subjects[valid].tolist()))})
            inner = make_splits(part.y[train], part.subjects[train], protocol,
                                config["inner_folds"], config["seed"] + fold + 1)
            write_json(fold_dir / "inner_splits.json", [
                {"train_rows": train[a].tolist(), "valid_rows": train[b].tolist()}
                for a, b in inner])
            for name in config["models"]:
                print(f"{protocol} fold {fold + 1}/{len(outer)}: {name}", flush=True)
                base, results, search_seconds = search(
                    name, config, part.X[train], part.y[train], inner)
                model_dir = fold_dir / name
                model_dir.mkdir()
                results.to_csv(model_dir / "search.csv", index=False)
                fixed, best, compact = candidates(results, config["compact_tolerance"])
                choices = {f"k_{k}": i for k, i in fixed.items()}
                choices.update(best=best, compact=compact)
                # Fit/evaluate each unique winner once, reuse its metrics for policy rows.
                cache = {}
                for choice, index in choices.items():
                    params = results.iloc[index].params
                    if index not in cache:
                        dest = model_dir / f"candidate_{index}"
                        model, seconds = fit(base, params, part.X[train], part.y[train], config["threads"])
                        values = evaluate(model, part.X[valid], part.y[valid], part.subjects[valid],
                                          valid, dataset.labels, dest, config["threads"])
                        values.update(fit_seconds=seconds, search_seconds=search_seconds)
                        save_features(model, dataset.features, dest)
                        write_json(dest / "params.json", params)
                        joblib.dump(model, dest / "model.joblib")
                        cache[index] = values
                    metrics.append({"protocol": protocol, "fold": fold, "model": name,
                                    "choice": choice, "k": params.get("select__k", 561),
                                    "candidate": index, **cache[index]})
                pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    table = pd.DataFrame(metrics)
    table.groupby(["protocol", "model", "choice"])[
        ["macro_f1", "accuracy", "fit_seconds", "prediction_us_per_row"]
    ].agg(["mean", "std"]).to_csv(output / "summary.csv")
    write_json(output / "complete.json", {"status": "complete", "official_test_used": False})
    return output


def freeze(data_root, config, output):
    """Select final best/compact configurations on all training subjects only."""
    validate_config(config)
    dataset = Dataset(data_root)
    part = dataset.load("train")
    output = prepare_output(output, config, dataset)
    cv = make_splits(part.y, part.subjects, "grouped", config["inner_folds"], config["seed"])
    write_json(output / "splits.json", [{"train": a.tolist(), "valid": b.tolist()} for a, b in cv])
    manifest = {"config": config, "train_fingerprint": dataset.fingerprint(), "models": []}
    for name in config["models"]:
        print(f"Freezing {name} using grouped training-only validation", flush=True)
        base, results, search_seconds = search(name, config, part.X, part.y, cv)
        results.to_csv(output / f"{name}_search.csv", index=False)
        _, best, compact = candidates(results, config["compact_tolerance"])
        for choice, index in [("best", best), ("compact", compact)]:
            params = results.iloc[index].params
            model, seconds = fit(base, params, part.X, part.y, config["threads"])
            model_id = f"{name}_{choice}"
            dest = output / model_id
            dest.mkdir()
            joblib.dump(model, dest / "model.joblib")
            save_features(model, dataset.features, dest)
            manifest["models"].append({"id": model_id, "model": name, "choice": choice,
                "params": params, "inner_macro_f1": float(results.iloc[index].mean_test_score),
                "fit_seconds": seconds, "search_seconds": search_seconds})
    write_json(output / "manifest.json", manifest)
    return output


def evaluate_final(data_root, frozen, output):
    import json
    frozen = Path(frozen)
    manifest = json.loads((frozen / "manifest.json").read_text())
    dataset = Dataset(data_root)
    if dataset.fingerprint() != manifest["train_fingerprint"]:
        raise ValueError("Training data changed since configurations were frozen")
    train = dataset.load("train")
    test = dataset.load("test")
    if set(train.subjects) & set(test.subjects):
        raise ValueError("Official partitions share participants")
    output = prepare_output(output, manifest["config"], dataset)
    write_json(output / "test_fingerprint.json", dataset.fingerprint("test"))
    rows = []
    for entry in manifest["models"]:
        model = joblib.load(frozen / entry["id"] / "model.joblib")
        values = evaluate(model, test.X, test.y, test.subjects, np.arange(len(test.y)),
                          dataset.labels, output / entry["id"], manifest["config"]["threads"])
        rows.append({**entry, **values})
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    write_json(output / "complete.json", {"status": "complete", "official_test_used": True,
                                          "frozen_manifest": manifest})
    return output

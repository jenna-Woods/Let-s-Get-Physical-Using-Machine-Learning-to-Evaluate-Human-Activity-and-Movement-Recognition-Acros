"""Scientific integrity tests using synthetic data; no official test access."""
import tempfile
import unittest
from unittest.mock import patch
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from har.experiments import candidates, search, run_cv, freeze, evaluate_final
from har.data import Partition, extract_safe
from har.models import build
from har.splits import make_splits


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(334)
        self.subjects = np.repeat(np.arange(12), 24)
        self.y = np.tile(np.repeat(np.arange(1, 7), 4), 12)
        self.X = rng.normal(size=(len(self.y), 30))
        self.X[:, 0] += self.y * 2

    def test_grouped_nested_disjoint_and_exhaustive(self):
        outer = make_splits(self.y, self.subjects, "grouped", 3, 334)
        seen = []
        for a, b in outer:
            self.assertFalse(set(self.subjects[a]) & set(self.subjects[b]))
            seen.extend(b)
            for c, d in make_splits(self.y[a], self.subjects[a], "grouped", 2, 335):
                self.assertFalse(set(self.subjects[a[c]]) & set(self.subjects[a[d]]))
                self.assertFalse(set(a[c]) & set(b))
        self.assertEqual(sorted(seen), list(range(len(self.y))))

    def test_selector_scaler_only_learn_training_and_roundtrip(self):
        base, _ = build("logistic", 334, True)
        model = clone(base).set_params(select__k=20).fit(self.X[:144], self.y[:144])
        expected = self.X[:144, model.named_steps["select"].get_support()].mean(axis=0)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, expected)
        scores = model.named_steps["select"].scores_.copy()
        model.predict(self.X[144:] + 10000)
        np.testing.assert_array_equal(scores, model.named_steps["select"].scores_)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(self.X), joblib.load(path).predict(self.X))

    def test_all_models_search_and_predict_six_classes(self):
        cv = make_splits(self.y, self.subjects, "grouped", 2, 334)
        config = {"seed": 334, "quick": True, "feature_counts": [10, 20],
                  "search_jobs": 1, "threads": 1}
        for name in ["dummy", "logistic", "forest", "svm"]:
            base, results, _ = search(name, config, self.X, self.y, cv)
            _, best, _ = candidates(results, .01)
            model = base.set_params(**results.iloc[best].params).fit(self.X, self.y)
            self.assertEqual(len(model.predict(self.X[:7])), 7)
            self.assertEqual(set(model.classes_), set(range(1, 7)))

    def test_compact_choice_uses_inner_scores_and_tolerance(self):
        table = pd.DataFrame({"params": [{"select__k": 20}, {"select__k": 50},
                                         {"select__k": 561}],
                              "mean_test_score": [.94, .965, .97]})
        fixed, best, compact = candidates(table, .01)
        self.assertEqual(best, 2)
        self.assertEqual(compact, 1)
        self.assertEqual(len(fixed), 3)

    def test_end_to_end_and_test_partition_isolation(self):
        train = Partition(self.X, self.y, self.subjects)
        test = Partition(self.X[:24], self.y[:24], np.full(24, 99))
        labels = {i: f"activity_{i}" for i in range(1, 7)}
        features = pd.DataFrame({"feature_id": range(1, 31),
                                 "name": [f"sensor-{i}" for i in range(30)]})
        loads = []

        class SyntheticDataset:
            def __init__(self, root):
                self.labels, self.features = labels, features

            def load(self, split):
                loads.append(split)
                return train if split == "train" else test

            def fingerprint(self, split="train"):
                return {split: "synthetic"}

        config = {"seed": 334, "outer_folds": 2, "inner_folds": 2,
                  "feature_counts": [10, 20], "models": ["dummy", "logistic"],
                  "protocols": ["grouped", "random"], "search_jobs": 1,
                  "threads": 1, "compact_tolerance": .01, "quick": True}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            with patch("har.experiments.Dataset", SyntheticDataset):
                run_cv("unused", config, root / "cv")
                freeze("unused", config, root / "frozen")
                self.assertNotIn("test", loads)
                evaluate_final("unused", root / "frozen", root / "final")
                self.assertEqual(loads.count("test"), 1)
            self.assertTrue((root / "cv" / "complete.json").exists())
            self.assertEqual(len(pd.read_csv(root / "final" / "metrics.csv")), 4)
            with self.assertRaises(ValueError):
                from har.experiments import prepare_output
                prepare_output(root / "cv", config, SyntheticDataset(None))

    def test_archive_path_traversal_rejected(self):
        import io
        import zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("../escape.txt", "bad")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            with zipfile.ZipFile(buffer) as z, self.assertRaises(ValueError):
                extract_safe(z, folder)


if __name__ == "__main__":
    unittest.main()

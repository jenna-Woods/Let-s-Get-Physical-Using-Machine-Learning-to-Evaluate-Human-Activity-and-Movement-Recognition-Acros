"""Load original feature files; keep subjects out of the design matrix."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def extract_safe(archive, destination):
    root = Path(destination).resolve()
    for entry in archive.infolist():
        target = (root / entry.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Archive entry escapes destination")
    archive.extractall(root)


def download(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "UCI HAR Dataset" / "features.txt").exists():
        return destination / "UCI HAR Dataset"
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        extract_safe(archive, destination)
    nested = destination / "UCI HAR Dataset.zip"
    if nested.exists():
        with zipfile.ZipFile(nested) as archive:
            extract_safe(archive, destination)
    if not (destination / "UCI HAR Dataset" / "features.txt").exists():
        raise ValueError("Downloaded archive does not contain the expected dataset")
    (destination / "download.json").write_text(json.dumps({
        "url": URL, "sha256": hashlib.sha256(payload).hexdigest()
    }, indent=2), encoding="utf-8")
    return destination / "UCI HAR Dataset"


@dataclass
class Partition:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        feature_rows = [line.split(maxsplit=1) for line in
                        (self.root / "features.txt").read_text().splitlines()]
        self.features = pd.DataFrame(feature_rows, columns=["feature_id", "name"])
        self.features["feature_id"] = self.features.feature_id.astype(int)
        if self.features.feature_id.tolist() != list(range(1, 562)):
            raise ValueError("Expected 561 ordered feature IDs")
        self.labels = dict((int(i), name) for i, name in
                           (line.split(maxsplit=1) for line in
                            (self.root / "activity_labels.txt").read_text().splitlines()))
        if set(self.labels) != set(range(1, 7)):
            raise ValueError("Expected six activities numbered 1-6")

    def load(self, split):
        if split not in {"train", "test"}:
            raise ValueError("Split must be train or test")
        folder = self.root / split
        part = Partition(
            np.loadtxt(folder / f"X_{split}.txt"),
            np.loadtxt(folder / f"y_{split}.txt", dtype=int, ndmin=1),
            np.loadtxt(folder / f"subject_{split}.txt", dtype=int, ndmin=1))
        expected = 7352 if split == "train" else 2947
        if part.X.shape != (expected, 561):
            raise ValueError(f"Unexpected {split} shape: {part.X.shape}")
        if len(part.y) != expected or len(part.subjects) != expected:
            raise ValueError("Features, labels, and subjects are misaligned")
        if not np.isfinite(part.X).all() or set(part.y) != set(self.labels):
            raise ValueError("Nonfinite features or unexpected activity labels")
        if not set(part.subjects).issubset(set(range(1, 31))):
            raise ValueError("Unexpected participant ID")
        expected_subjects = 21 if split == "train" else 9
        if len(np.unique(part.subjects)) != expected_subjects:
            raise ValueError("Unexpected number of participants")
        return part

    def fingerprint(self, split="train"):
        paths = [self.root / "features.txt", self.root / "activity_labels.txt"]
        paths += [self.root / split / f"{name}_{split}.txt"
                  for name in ["X", "y", "subject"]]
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths}


def audit(dataset, part, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.crosstab(part.subjects, part.y).to_csv(output / "subject_activity_counts.csv")
    summary = pd.DataFrame({"mean": part.X.mean(axis=0),
                            "std": part.X.std(axis=0),
                            "min": part.X.min(axis=0),
                            "max": part.X.max(axis=0)})
    pd.concat([dataset.features, summary], axis=1).to_csv(output / "features.csv", index=False)
    corr = np.corrcoef(part.X, rowvar=False)
    a, b = np.where(np.triu(np.abs(corr) > .95, k=1))
    pd.DataFrame({"feature_id_a": a + 1, "feature_id_b": b + 1,
                  "correlation": corr[a, b]}).to_csv(output / "correlated_pairs.csv", index=False)
    (output / "audit.json").write_text(json.dumps({
        "shape": list(part.X.shape), "participants": len(np.unique(part.subjects)),
        "missing_values": int(np.isnan(part.X).sum()),
        "constant_features": int((part.X.std(axis=0) == 0).sum()),
        "fingerprint": dataset.fingerprint()
    }, indent=2), encoding="utf-8")

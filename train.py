"""Train five reproducible classifiers once, then save models and holdout results.

No hyperparameters are selected using the holdout. Duplicate feature vectors are
removed *before* splitting; scaler statistics are fitted on training rows only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fetch_data import fetch_dataset, sha256_file
from ml_core import (
    CLASS_LABELS, FEATURES, METRIC_NAMES, MODEL_FILES, TARGET,
    evaluate_model, validate_test_data,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
PROJECT_DIR = Path(__file__).resolve().parent


def make_models() -> dict[str, Pipeline]:
    """Return untrained, modest baseline models with explicitly recorded settings."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=4, random_state=RANDOM_STATE
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=9, weights="uniform")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=RANDOM_STATE)),
        ]),
    }


def class_counts(data: pd.DataFrame) -> dict[str, int]:
    """Keep both class labels and a stable ordering in provenance files."""
    return {label: int(data[TARGET].eq(label).sum()) for label in CLASS_LABELS}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(
    data_dir: Path = PROJECT_DIR / "data",
    model_dir: Path = PROJECT_DIR / "model",
    results_dir: Path = PROJECT_DIR / "results",
    test_output: Path = PROJECT_DIR / "test_data.csv",
) -> pd.DataFrame:
    """Train, persist full pipelines, and evaluate one untouched stratified holdout."""
    raw_path = fetch_dataset(data_dir)
    raw = pd.read_csv(raw_path, sep=";", header=None, names=FEATURES + [TARGET])
    raw = validate_test_data(raw)
    # Conflicting labels must be investigated rather than silently discarded.
    label_counts = raw.groupby(FEATURES, sort=False, dropna=False)[TARGET].nunique()
    conflicts = int(label_counts.gt(1).sum())
    if conflicts:
        raise ValueError(f"Found {conflicts} identical feature vectors with conflicting class labels.")
    unique = raw.drop_duplicates(subset=FEATURES, keep="first")
    dropped_indices = raw.index[~raw.index.isin(unique.index)].tolist()
    train_indices, test_indices = train_test_split(
        unique.index.to_numpy(), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=unique[TARGET],
    )
    training = unique.loc[train_indices]
    holdout = unique.loc[test_indices]
    for directory in (Path(model_dir), Path(results_dir), Path(test_output).parent):
        directory.mkdir(parents=True, exist_ok=True)
    # Public evaluation sample; training source rows remain in original UCI data.
    holdout.to_csv(test_output, index=False)

    models = make_models()
    evaluations = {}
    metric_rows = []
    saved_hashes = {}
    for name, model in models.items():
        model.fit(training[FEATURES], training[TARGET])
        artifact = Path(model_dir) / MODEL_FILES[name]
        joblib.dump(model, artifact, compress=3)
        saved_hashes[MODEL_FILES[name]] = sha256_file(artifact)
        result = evaluate_model(model, holdout)
        evaluations[name] = {
            "metrics": result["metrics"],
            "confusion_matrix": result["confusion_matrix"],
            "classification_report": result["classification_report"],
        }
        metric_rows.append({"ML Model Name": name, **result["metrics"]})
        print(f"Trained {name}: accuracy={result['metrics']['Accuracy']:.4f}, "
              f"AUC={result['metrics']['AUC']:.4f}, MCC={result['metrics']['MCC']:.4f}")
    metrics = pd.DataFrame(metric_rows, columns=["ML Model Name", *METRIC_NAMES])
    metrics.to_csv(Path(results_dir) / "metrics.csv", index=False)
    write_json(Path(results_dir) / "evaluation.json", {
        "positive_class": "RB", "confusion_matrix_order": CLASS_LABELS,
        "test_rows": len(holdout), "models": evaluations,
    })
    split_indices = {
        "train_indices": [int(index) for index in train_indices],
        "test_indices": [int(index) for index in test_indices],
    }
    split_hash = hashlib.sha256(
        json.dumps(split_indices, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "UCI QSAR biodegradation", "uci_id": 254,
            "source_file": "data/biodeg.csv", "source_sha256": sha256_file(raw_path),
            "raw_rows": len(raw), "feature_count": len(FEATURES),
            "deduplicated_rows": len(unique), "duplicate_rows_removed": len(dropped_indices),
            "dropped_original_indices": [int(index) for index in dropped_indices],
            "conflicting_feature_groups": conflicts,
            "class_distribution_raw": class_counts(raw),
            "class_distribution_cleaned": class_counts(unique),
        },
        "features": FEATURES, "target": TARGET, "class_labels": CLASS_LABELS,
        "positive_class": "RB",
        "split": {
            **split_indices,
            "index_reference": "Zero-based ORIGINAL data/biodeg.csv row indices, before deduplication",
            "train_rows": len(training), "test_rows": len(holdout),
            "class_distribution_train": class_counts(training),
            "class_distribution_test": class_counts(holdout),
            "indices_sha256": split_hash,
            "test_csv_sha256": sha256_file(Path(test_output)),
        },
        "settings": {
            "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
            "stratify": TARGET,
            "duplicate_policy": "Keep first identical feature vector after checking label consistency, before splitting",
            "scaler_fit": "Training partition only, within each saved pipeline",
            "tuning": "No hyperparameter search; holdout used only for final evaluation",
            "models": {
                name: {
                    "file": MODEL_FILES[name],
                    "scaling": "StandardScaler" if "scaler" in model.named_steps else "None",
                    "classifier": model.named_steps["classifier"].__class__.__name__,
                    "parameters": model.named_steps["classifier"].get_params(),
                } for name, model in models.items()
            },
        },
        "versions": {
            "python": platform.python_version(), "scikit-learn": sklearn_version,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
            "streamlit": importlib.metadata.version("streamlit"),
        },
        "model_sha256": saved_hashes,
    }
    write_json(Path(results_dir) / "manifest.json", manifest)
    print(f"Split: {len(raw)} raw -> {len(unique)} unique -> "
          f"{len(training)} training + {len(holdout)} test rows.")
    print(f"Saved models to {model_dir}; results to {results_dir}; test data to {test_output}.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "model")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--test-output", type=Path, default=PROJECT_DIR / "test_data.csv")
    arguments = parser.parse_args()
    train(arguments.data_dir, arguments.model_dir, arguments.results_dir, arguments.test_output)


if __name__ == "__main__":
    main()

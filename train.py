"""Train and evaluate the five QSAR classifiers as a separate offline step.

The workflow verifies the source data, removes duplicate feature vectors, makes
one stratified train/test split, and saves each complete fitted pipeline. Every
model uses the same split and fixed settings. The holdout is reserved for final
evaluation: neither hyperparameter selection nor scaler fitting uses its rows.

Outputs include the labeled test CSV, five model files, six-metric comparison,
confusion matrices, classification reports, and a reproducibility manifest.
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
    CLASS_LABELS,
    FEATURES,
    METRIC_NAMES,
    MODEL_FILES,
    TARGET,
    evaluate_model,
    validate_test_data,
)

# A shared seed and class-stratified split make model comparisons reproducible.
RANDOM_STATE = 42
TEST_SIZE = 0.2
PROJECT_DIR = Path(__file__).resolve().parent


def make_models() -> dict[str, Pipeline]:
    """Build the five unfitted baselines without consulting the test partition.

    StandardScaler is included in the applicable pipelines so its training-only
    transformation is also used during deployment. Tree thresholds do not need
    standardized inputs, so the decision tree uses the original feature scale.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=1.0, max_iter=3000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=9, weights="uniform")),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def class_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count NRB and RB in a stable order, including any zero-count class."""
    return {label: int(data[TARGET].eq(label).sum()) for label in CLASS_LABELS}


def write_json(path: Path, value: dict) -> None:
    """Save readable, strict JSON for evaluation and reproducibility records."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(
    data_dir: Path = PROJECT_DIR / "data",
    model_dir: Path = PROJECT_DIR / "model",
    results_dir: Path = PROJECT_DIR / "results",
    test_output: Path = PROJECT_DIR / "test_data.csv",
) -> pd.DataFrame:
    """Generate fitted pipelines and their results on one shared holdout.

    Original row indices are retained through deduplication and splitting. The
    manifest records these indices so the train/test membership can be audited
    against the unmodified UCI file. Return the six-metric comparison table.
    """
    # Verify the original bytes before assigning UCI's descriptor names. The
    # shared validator checks numeric inputs, finite values, and class labels.
    raw_path = fetch_dataset(data_dir)
    raw = pd.read_csv(raw_path, sep=";", header=None, names=FEATURES + [TARGET])
    raw = validate_test_data(raw)
    # Audit labels before removing repeated feature vectors. If a duplicate
    # vector had conflicting labels, dropping it would hide a data quality issue.
    label_counts = raw.groupby(FEATURES, sort=False, dropna=False)[TARGET].nunique()
    conflicts = int(label_counts.gt(1).sum())
    if conflicts:
        raise ValueError(
            f"Found {conflicts} identical feature vectors with conflicting class labels."
        )
    unique = raw.drop_duplicates(subset=FEATURES, keep="first")
    dropped_indices = raw.index[~raw.index.isin(unique.index)].tolist()

    # Deduplicate before splitting so an identical chemical descriptor vector
    # cannot appear in both partitions. Stratification preserves class balance.
    train_indices, test_indices = train_test_split(
        unique.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=unique[TARGET],
    )
    training = unique.loc[train_indices]
    holdout = unique.loc[test_indices]
    for directory in (Path(model_dir), Path(results_dir), Path(test_output).parent):
        directory.mkdir(parents=True, exist_ok=True)

    # Publish only the holdout as the app's test sample, with all feature names
    # and the target. The original source file still contains the training rows.
    holdout.to_csv(test_output, index=False)

    models = make_models()
    evaluations = {}
    metric_rows = []
    saved_hashes = {}
    for name, model in models.items():
        # Pipeline.fit learns both scaler statistics and classifier parameters
        # from training data only. Saving the whole pipeline preserves this order.
        model.fit(training[FEATURES], training[TARGET])
        artifact = Path(model_dir) / MODEL_FILES[name]
        joblib.dump(model, artifact, compress=3)
        saved_hashes[MODEL_FILES[name]] = sha256_file(artifact)

        # Evaluation predicts with the fitted pipeline; it never fits or tunes it.
        result = evaluate_model(model, holdout)
        evaluations[name] = {
            "metrics": result["metrics"],
            "confusion_matrix": result["confusion_matrix"],
            "classification_report": result["classification_report"],
        }
        metric_rows.append({"ML Model Name": name, **result["metrics"]})
        print(
            f"Trained {name}: accuracy={result['metrics']['Accuracy']:.4f}, "
            f"AUC={result['metrics']['AUC']:.4f}, MCC={result['metrics']['MCC']:.4f}"
        )

    # Save full-precision results; display rounding belongs in the app/README.
    metrics = pd.DataFrame(metric_rows, columns=["ML Model Name", *METRIC_NAMES])
    metrics.to_csv(Path(results_dir) / "metrics.csv", index=False)
    write_json(
        Path(results_dir) / "evaluation.json",
        {
            "positive_class": "RB",
            "confusion_matrix_order": CLASS_LABELS,
            "test_rows": len(holdout),
            "models": evaluations,
        },
    )

    # Record exact partition membership, not just the seed. Checksums connect
    # the source data, exported test CSV, and saved pipelines to this run.
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
            "name": "UCI QSAR biodegradation",
            "uci_id": 254,
            "source_file": "data/biodeg.csv",
            "source_sha256": sha256_file(raw_path),
            "raw_rows": len(raw),
            "feature_count": len(FEATURES),
            "deduplicated_rows": len(unique),
            "duplicate_rows_removed": len(dropped_indices),
            "dropped_original_indices": [int(index) for index in dropped_indices],
            "conflicting_feature_groups": conflicts,
            "class_distribution_raw": class_counts(raw),
            "class_distribution_cleaned": class_counts(unique),
        },
        "features": FEATURES,
        "target": TARGET,
        "class_labels": CLASS_LABELS,
        "positive_class": "RB",
        "split": {
            **split_indices,
            "index_reference": (
                "Zero-based ORIGINAL data/biodeg.csv row indices, before deduplication"
            ),
            "train_rows": len(training),
            "test_rows": len(holdout),
            "class_distribution_train": class_counts(training),
            "class_distribution_test": class_counts(holdout),
            "indices_sha256": split_hash,
            "test_csv_sha256": sha256_file(Path(test_output)),
        },
        "settings": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "stratify": TARGET,
            "duplicate_policy": (
                "Keep first identical feature vector after checking label consistency, "
                "before splitting"
            ),
            "scaler_fit": "Training partition only, within each saved pipeline",
            "tuning": "No hyperparameter search; holdout used only for final evaluation",
            "models": {
                name: {
                    "file": MODEL_FILES[name],
                    "scaling": ("StandardScaler" if "scaler" in model.named_steps else "None"),
                    "classifier": model.named_steps["classifier"].__class__.__name__,
                    "parameters": model.named_steps["classifier"].get_params(),
                }
                for name, model in models.items()
            },
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "streamlit": importlib.metadata.version("streamlit"),
        },
        "model_sha256": saved_hashes,
    }
    write_json(Path(results_dir) / "manifest.json", manifest)
    print(
        f"Split: {len(raw)} raw -> {len(unique)} unique -> "
        f"{len(training)} training + {len(holdout)} test rows."
    )
    print(f"Saved models to {model_dir}; results to {results_dir}; test data to {test_output}.")
    return metrics


def main() -> None:
    """Read optional output paths and run the offline training workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "model")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--test-output", type=Path, default=PROJECT_DIR / "test_data.csv")
    arguments = parser.parse_args()
    train(
        arguments.data_dir,
        arguments.model_dir,
        arguments.results_dir,
        arguments.test_output,
    )


if __name__ == "__main__":
    main()

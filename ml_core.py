"""Shared schema, model loading, and holdout evaluation for QSAR biodegradation.

The target is ready biodegradability: RB is the positive class, NRB the negative
class. Models are trained separately by ``train.py``; this module never fits one.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

# Descriptor names and order published by UCI dataset 254. The original CSV has
# no header; adding these names makes uploaded test data understandable.
FEATURES: list[str] = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)",
    "nCIR", "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d",
    "B04[C-Br]", "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon",
    "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX",
]
TARGET = "class"
CLASS_LABELS = ["NRB", "RB"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def validate_test_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a labeled test frame and return columns in the training order.

    Headers and class labels may have surrounding whitespace. Missing/extra
    columns, duplicate headers, missing values, nonnumeric/infinite descriptors,
    empty data, and labels other than NRB/RB raise a readable ``ValueError``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Test data must be a pandas DataFrame.")
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    duplicated = data.columns[data.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate column names: {', '.join(duplicated)}.")
    expected = FEATURES + [TARGET]
    missing = [column for column in expected if column not in data.columns]
    extra = [column for column in data.columns if column not in expected]
    if missing or extra:
        messages = []
        if missing:
            messages.append("Missing columns: " + ", ".join(missing))
        if extra:
            messages.append("Unexpected columns: " + ", ".join(extra))
        raise ValueError("; ".join(messages) + ". Upload the supplied test CSV format.")
    if data.empty:
        raise ValueError("The test CSV contains no data rows.")
    data = data.loc[:, expected]
    if data.isna().any().any():
        raise ValueError("The test CSV contains missing values; every cell is required.")
    for column in FEATURES:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature '{column}' must contain numeric values only.") from error
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("All feature values must be finite numbers; NaN and infinity are invalid.")
    data[TARGET] = data[TARGET].astype(str).str.strip()
    unknown = sorted(set(data[TARGET]) - set(CLASS_LABELS))
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}. Use only NRB or RB.")
    return data


def read_test_csv(source: Any) -> pd.DataFrame:
    """Read UTF-8 comma-separated test CSV from bytes, a path, or a file object.

    The standard CSV parser checks the *original* header before pandas can
    rename duplicate columns. Every record must have exactly the header width.
    """
    try:
        if isinstance(source, bytes):
            content = source.decode("utf-8-sig")
        elif isinstance(source, (str, Path)):
            content = Path(source).read_text(encoding="utf-8-sig")
        elif hasattr(source, "read"):
            raw = source.read()
            content = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        else:
            raise ValueError("Provide CSV bytes, a file path, or a readable file.")
    except (UnicodeDecodeError, OSError) as error:
        raise ValueError("The test file must be an accessible UTF-8 CSV.") from error
    try:
        reader = csv.reader(io.StringIO(content), strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError("The CSV is empty; include the header and labeled test rows.")
        header = [column.strip() for column in header]
        if len(set(header)) != len(header):
            raise ValueError("Duplicate column names are not allowed in a test CSV.")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV record {line_number} has {len(row)} fields; expected {len(header)}."
                )
            rows.append(row)
    except csv.Error as error:
        raise ValueError(f"The CSV could not be parsed: {error}.") from error
    return validate_test_data(pd.DataFrame(rows, columns=header))


def load_models(model_dir: Path) -> dict[str, Pipeline]:
    """Load only the five fixed, project-generated pipeline files.

    Joblib uses pickle internally: these artifacts must come from our training
    run. The web app must never accept uploaded model files.
    """
    directory = Path(model_dir)
    models = {}
    for name, filename in MODEL_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained model: {path}. Run python train.py first.")
        model = joblib.load(path)
        if not isinstance(model, Pipeline):
            raise ValueError(f"{filename} is not a saved scikit-learn pipeline.")
        if list(model.feature_names_in_) != FEATURES:
            raise ValueError(f"The feature schema in {filename} does not match this app.")
        if set(model.classes_) != set(CLASS_LABELS):
            raise ValueError(f"The class labels in {filename} do not match this app.")
        models[name] = model
    return models


def evaluate_model(model: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate an already fitted model; return all six metrics and diagnostics.

    AUC uses the RB probability or continuous decision score. AUC is undefined
    for a one-class upload and is then ``None``; the other metrics still work.
    Confusion matrix axes always follow [NRB, RB], even for one-class uploads.
    """
    data = validate_test_data(frame)
    features = data[FEATURES]
    actual = data[TARGET]
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index("RB")
        scores = model.predict_proba(features)[:, positive_index]
    else:
        scores = np.asarray(model.decision_function(features))
        if list(model.classes_)[1] != "RB":
            scores = -scores
    metrics = {
        "Accuracy": float(accuracy_score(actual, predictions)),
        "AUC": float(roc_auc_score(actual.eq("RB"), scores)) if actual.nunique() == 2 else None,
        "Precision": float(precision_score(actual, predictions, pos_label="RB", zero_division=0)),
        "Recall": float(recall_score(actual, predictions, pos_label="RB", zero_division=0)),
        "F1": float(f1_score(actual, predictions, pos_label="RB", zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predictions)),
    }
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(actual, predictions, labels=CLASS_LABELS).tolist(),
        "classification_report": classification_report(
            actual, predictions, labels=CLASS_LABELS, output_dict=True, zero_division=0
        ),
        "predictions": np.asarray(predictions),
        "scores": np.asarray(scores),
    }


def evaluate_models(models: dict[str, Pipeline], frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the five saved models on the same labeled test data."""
    return pd.DataFrame([
        {"ML Model Name": name, **evaluate_model(model, frame)["metrics"]}
        for name, model in models.items()
    ], columns=["ML Model Name", *METRIC_NAMES])

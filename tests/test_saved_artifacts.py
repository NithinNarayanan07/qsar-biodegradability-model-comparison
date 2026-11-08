"""Check reproducibility and split isolation using the shipped source dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ml_core import FEATURES, MODEL_FILES, TARGET, load_models, read_test_csv


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts():
    raw = pd.read_csv(ROOT / "data" / "biodeg.csv", sep=";", header=None, names=FEATURES + [TARGET])
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
    models = load_models(ROOT / "model")
    test_frame = read_test_csv(ROOT / "test_data.csv")
    return raw, manifest, models, test_frame


def test_holdout_rows_are_disjoint_from_training_feature_vectors(artifacts):
    raw, manifest, _, test_frame = artifacts
    train_indices = manifest["split"]["train_indices"]
    test_indices = manifest["split"]["test_indices"]
    assert len(raw) == 1055
    assert len(FEATURES) == 41
    assert not set(train_indices) & set(test_indices)
    train_vectors = set(raw.loc[train_indices, FEATURES].itertuples(index=False, name=None))
    test_vectors = set(raw.loc[test_indices, FEATURES].itertuples(index=False, name=None))
    assert not train_vectors & test_vectors
    expected = raw.loc[test_indices, FEATURES + [TARGET]].reset_index(drop=True)
    pd.testing.assert_frame_equal(test_frame.reset_index(drop=True), expected, check_dtype=False)
    assert set(raw.loc[train_indices, TARGET]) == {"NRB", "RB"}
    assert set(test_frame[TARGET]) == {"NRB", "RB"}


def test_saved_scalers_were_fitted_only_on_the_training_partition(artifacts):
    raw, manifest, models, _ = artifacts
    train_frame = raw.loc[manifest["split"]["train_indices"], FEATURES]
    scaler_count = 0
    for model in models.values():
        for _, step in model.steps:
            if isinstance(step, StandardScaler):
                scaler_count += 1
                np.testing.assert_allclose(step.mean_, train_frame.mean().to_numpy(), rtol=1e-12, atol=1e-12)
                assert np.all(np.asarray(step.n_samples_seen_) == len(train_frame))
    assert scaler_count == 4  # The tree keeps the original scale.


def test_reloaded_models_reproduce_the_committed_comparison_metrics(artifacts):
    _, _, models, test_frame = artifacts
    saved = pd.read_csv(ROOT / "results" / "metrics.csv").set_index("ML Model Name")
    assert list(saved.index) == list(MODEL_FILES)
    actual = test_frame[TARGET]
    for name, model in models.items():
        predicted = model.predict(test_frame[FEATURES])
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(test_frame[FEATURES])[:, list(model.classes_).index("RB")]
        else:
            scores = model.decision_function(test_frame[FEATURES])
            if list(model.classes_)[1] != "RB":
                scores = -scores
        expected = {
            "Accuracy": accuracy_score(actual, predicted),
            "AUC": roc_auc_score(actual.eq("RB"), scores),
            "Precision": precision_score(actual, predicted, pos_label="RB", zero_division=0),
            "Recall": recall_score(actual, predicted, pos_label="RB", zero_division=0),
            "F1": f1_score(actual, predicted, pos_label="RB", zero_division=0),
            "MCC": matthews_corrcoef(actual, predicted),
        }
        for metric, value in expected.items():
            assert saved.loc[name, metric] == pytest.approx(value, abs=1e-10), f"{name}: {metric}"

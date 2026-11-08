"""Verify the evaluator displays real artifacts without fitting a model."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from ml_core import METRIC_NAMES, MODEL_FILES, evaluate_model, load_models, read_test_csv
from presentation import MODEL_DISPLAY_NAMES


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("The evaluation app must never train a model.")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)


@pytest.fixture
def app():
    return AppTest.from_file(ROOT / "app.py", default_timeout=30).run()


def test_sample_loads_all_required_metrics_and_all_five_model_rows(app):
    assert not app.exception
    assert not app.error
    assert [metric.label for metric in app.metric] == METRIC_NAMES
    comparison = app.dataframe[0].value
    assert isinstance(comparison, pd.DataFrame)
    assert comparison.index.tolist() == [MODEL_DISPLAY_NAMES[name] for name in MODEL_FILES]
    assert comparison.columns.tolist() == METRIC_NAMES


def test_classifier_selection_updates_metrics_from_the_saved_model(app):
    models = load_models(ROOT / "model")
    test_frame = read_test_csv(ROOT / "test_data.csv")
    for selected in ["Naive Bayes", "Support Vector Machine"]:
        app.selectbox[0].select(selected).run()
        assert not app.exception
        assert not app.error
        assert MODEL_DISPLAY_NAMES[selected] in [heading.value for heading in app.subheader]
        expected = evaluate_model(models[selected], test_frame)["metrics"]
        assert {metric.label: metric.value for metric in app.metric} == {
            name: f"{expected[name]:.4f}" for name in METRIC_NAMES
        }


def test_upload_mode_requires_a_file_and_clears_sample_metrics(app):
    app.radio[0].set_value("Upload test CSV").run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 0
    assert len(app.info) == 1
    assert "Upload" in app.info[0].value


def upload_app(monkeypatch, payload):
    # AppTest does not expose a file-uploader driver. Substitute only the
    # widget's return value; the app still parses and evaluates the real CSV.
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(payload))
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Upload test CSV").run()
    return app


def test_valid_upload_is_parsed_and_compared_without_training(monkeypatch):
    frame = read_test_csv(ROOT / "test_data.csv").iloc[:25]
    app = upload_app(monkeypatch, frame.to_csv(index=False).encode())
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 6
    assert app.dataframe[0].value.shape == (5, 6)
    assert any("Uploaded test data" in caption.value and "25 chemical records" in caption.value for caption in app.caption)


def test_bad_upload_shows_validation_message_instead_of_results(monkeypatch):
    app = upload_app(monkeypatch, b"unrelated_column\n123\n")
    assert not app.exception
    assert len(app.error) == 1
    assert "Cannot evaluate this CSV" in app.error[0].value
    assert len(app.metric) == 0


def test_one_class_upload_keeps_metrics_but_explains_undefined_auc(monkeypatch):
    frame = read_test_csv(ROOT / "test_data.csv")
    frame = frame.loc[frame["class"] == "RB"].iloc[:10]
    app = upload_app(monkeypatch, frame.to_csv(index=False).encode())
    assert not app.exception
    assert not app.error
    assert {metric.label: metric.value for metric in app.metric}["AUC"] == "N/A"
    assert any("one class" in warning.value for warning in app.warning)
    assert app.dataframe[0].value.shape == (5, 6)

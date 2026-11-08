"""Check metric meaning independently of the training implementation."""

import math

import numpy as np
import pandas as pd
import pytest

from ml_core import FEATURES, TARGET, evaluate_model, evaluate_models


def example_frame():
    frame = pd.DataFrame(np.zeros((4, len(FEATURES))), columns=FEATURES)
    frame[FEATURES[0]] = np.arange(4)
    frame[TARGET] = ["NRB", "RB", "RB", "NRB"]
    return frame


class ReversedProbabilityModel:
    """RB is deliberately the first probability column, not the second."""

    classes_ = np.array(["RB", "NRB"])
    predictions = np.array(["NRB", "RB", "NRB", "NRB"])
    rb_scores = np.array([0.1, 0.9, 0.4, 0.8])

    def _rows(self, features):
        return features[FEATURES[0]].to_numpy(dtype=int)

    def predict(self, features):
        return self.predictions[self._rows(features)]

    def predict_proba(self, features):
        scores = self.rb_scores[self._rows(features)]
        return np.column_stack([scores, 1 - scores])


class ReversedDecisionModel:
    """Binary decision scores favor classes_[1], NRB in this fixture."""

    classes_ = np.array(["RB", "NRB"])

    def predict(self, features):
        return ReversedProbabilityModel().predict(features)

    def decision_function(self, features):
        rows = features[FEATURES[0]].to_numpy(dtype=int)
        return 0.5 - ReversedProbabilityModel.rb_scores[rows]


@pytest.mark.parametrize("model", [ReversedProbabilityModel(), ReversedDecisionModel()])
def test_metrics_use_rb_as_positive_and_rank_scores_for_auc(model):
    result = evaluate_model(model, example_frame())
    # TP=1, TN=2, FP=0, FN=1. Of four positive-negative pairs,
    # three have the positive example ranked higher, hence AUC=3/4.
    assert result["metrics"] == pytest.approx(
        {
            "Accuracy": 0.75,
            "AUC": 0.75,
            "Precision": 1.0,
            "Recall": 0.5,
            "F1": 2 / 3,
            "MCC": 2 / math.sqrt(12),
        }
    )
    assert np.asarray(result["confusion_matrix"]).tolist() == [[2, 0], [1, 1]]
    assert result["classification_report"]["RB"]["recall"] == 0.5
    assert result["classification_report"]["NRB"]["support"] == 2
    assert np.asarray(result["predictions"]).tolist() == ["NRB", "RB", "NRB", "NRB"]


def test_single_class_upload_has_undefined_auc_and_fixed_confusion_axes():
    frame = example_frame().iloc[[0, 3]]
    result = evaluate_model(ReversedProbabilityModel(), frame)
    assert result["metrics"]["AUC"] is None
    assert result["metrics"]["Accuracy"] == 1
    assert np.asarray(result["confusion_matrix"]).tolist() == [[2, 0], [0, 0]]
    assert all(
        np.isfinite(value)
        for key, value in result["metrics"].items()
        if key != "AUC"
    )


def test_comparison_evaluates_each_model_on_the_same_rows():
    models = {
        "Probability fixture": ReversedProbabilityModel(),
        "Decision fixture": ReversedDecisionModel(),
    }
    comparison = evaluate_models(models, example_frame())
    assert comparison["ML Model Name"].tolist() == list(models)
    assert comparison["AUC"].tolist() == pytest.approx([0.75, 0.75])
    assert comparison["MCC"].tolist() == pytest.approx([2 / math.sqrt(12)] * 2)

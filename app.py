"""Streamlit entry point for the five-classifier biodegradability assignment.

Faculty reading route:
1. main() shows the complete evaluation workflow at the end of this file.
2. Named render_* functions below correspond to the visible app sections.
3. ml_core.py owns validation and metrics; presentation.py owns visual styling.

Training is deliberately separate in train.py. This app only loads fitted
pipelines and evaluates the selected labeled test data; it never calls fit().
"""

import json
import os
from pathlib import Path
from typing import Any

# Set the plotting cache location before importing Matplotlib, including in labs.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

# Render figures without requiring a desktop display on Streamlit or the lab.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

from ml_core import (
    FEATURES,
    MODEL_FILES,
    TARGET,
    evaluate_model,
    evaluate_models,
    load_models,
    read_test_csv,
)
from presentation import (
    METRIC_HELP,
    MODEL_DISPLAY_NAMES,
    apply_theme,
    confusion_figure,
    render_hero,
    render_metrics,
    render_sidebar_brand,
    roc_figure,
    sidebar_step,
)

MODEL_NOTES = {
    "Logistic Regression": "A linear baseline. Molecular properties are standardized using the training data before classification.",
    "Decision Tree": "A sequence of decisions based on molecular-property thresholds. Properties are used on their original scales.",
    "KNN": "K-Nearest Neighbours (KNN) predicts by voting among the 9 closest training records. Standardization keeps large numerical scales from dominating distance.",
    "Naive Bayes": "Gaussian Naive Bayes models the distribution of each property within each class and assumes properties are conditionally independent given the class.",
    "Support Vector Machine": "Support Vector Machine (SVM) uses a radial basis function (RBF) kernel to learn a nonlinear boundary between the two classes.",
}


@st.cache_resource(show_spinner="Loading trained models...")
def saved_models() -> dict[str, Pipeline]:
    """Load only the five artifacts shipped with this project, never user files."""
    return load_models(ROOT / "model")


# Input controls and data validation


def render_sidebar(sample_path: Path) -> tuple[str, Any, str]:
    """Collect user selections without reading, saving or training on uploads."""
    with st.sidebar:
        render_sidebar_brand()
        sidebar_step("01", "Test data")
        source = st.radio("Test data source", ["Included test set", "Upload test CSV"])
        if sample_path.exists():
            st.download_button(
                "Download test CSV",
                sample_path.read_bytes(),
                "test_data.csv",
                "text/csv",
                width="stretch",
                type="primary",
            )
        st.caption(
            "Biodegradability data only. Match the example's 41 molecular-property columns and class labels."
        )
        st.caption("CSV = comma-separated values. Maximum size: 5 MB (megabytes).")
        uploaded = st.file_uploader(
            "Upload biodegradability CSV", type=["csv"], disabled=source == "Included test set"
        )
        sidebar_step("02", "Classifier")
        selected = st.selectbox(
            "Classifier", list(MODEL_FILES), format_func=MODEL_DISPLAY_NAMES.get
        )
        st.divider()
        st.html(
            '<div class="class-legend"><div><b>RB</b><span>Ready biodegradable</span></div><div class="nrb"><b>NRB</b><span>Not ready biodegradable</span></div></div>'
        )
        st.caption(
            "Models are trained offline. Uploads are used for evaluation only and are not saved to disk."
        )
        st.caption("Results update when you change the classifier or test data.")
    return source, uploaded, selected


def read_selected_test_data(source: str, uploaded: Any, sample_path: Path) -> pd.DataFrame:
    """Read one labeled test file; stop the page with guidance for invalid input."""
    if source == "Upload test CSV" and uploaded is None:
        st.info(
            "Upload a labeled biodegradability test CSV to compare all five classifiers. It must contain the same 41 molecular-property columns and known RB/NRB labels. Download the included example for the required format; unrelated datasets are not supported."
        )
        st.stop()
    try:
        frame = read_test_csv(
            uploaded.getvalue()
            if uploaded is not None and source == "Upload test CSV"
            else sample_path
        )
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(f"Cannot evaluate this CSV: {exc}")
        st.info(
            "This app accepts labeled chemical biodegradability data only. Download the example CSV and keep its 41 input column names plus the class column."
        )
        st.stop()
    return frame


# Faculty-facing data context and evaluation sections


def render_test_data_context(frame: pd.DataFrame, source: str) -> None:
    """Explain the active file, known labels and the original holdout split."""
    counts = frame[TARGET].value_counts()
    with st.container(key="data_context"):
        data_label = "Provided test data" if source == "Included test set" else "Uploaded test data"
        st.caption(
            f"{data_label} · {len(frame):,} chemical records · {len(FEATURES)} input properties  \n"
            f"Known labels: {counts.get('NRB', 0)} not ready biodegradable (NRB) · "
            f"{counts.get('RB', 0)} ready biodegradable (RB)"
        )
    with st.expander("About the test data"):
        st.markdown(
            "**What does each record contain?** Each row represents one chemical. "
            "The 41 numerical input columns describe its molecular properties. "
            "The additional `class` column gives the known answer: "
            "**ready biodegradable (RB)** or **not ready biodegradable (NRB)**."
        )
        st.markdown(
            "**How was the provided test set prepared?** The original dataset contains "
            "1,055 records. Removing 3 duplicate rows leaves 1,052 records: "
            "**841 for training (about 80%)** and **211 for testing (about 20%)**. "
            "Those 211 test records were kept separate from model training."
        )
        st.markdown(
            "**How are results evaluated?** All five models receive the same selected "
            "test records. Their predictions are compared with the known labels to "
            "calculate the evaluation metrics. The category counts above describe the "
            "known labels in the selected file."
        )
        st.markdown(
            "**Using your own test file?** This app evaluates chemical biodegradability "
            "only. Include the same 41 molecular-property column names and "
            "the `class` column, using `RB` or `NRB` as label values. Columns may appear "
            "in any order. Download the provided test CSV as an example. Uploaded files "
            "are evaluated using the saved models without retraining; use records that "
            "were not used to train the models."
        )
        st.markdown(
            "**What is checked during upload?** The app checks column names, numeric "
            "values, missing values and class labels. It cannot verify whether the "
            "numbers represent genuine molecular measurements or whether the records "
            "were used in training. Renaming an unrelated dataset's columns does not "
            "make it suitable for these models."
        )


def render_metric_guide(frame: pd.DataFrame) -> None:
    """Expand metric terminology and explain the single-class AUC limitation."""
    st.caption(
        "AUC = area under the receiver operating characteristic (ROC) curve. F1 = precision–recall balance. MCC = Matthews correlation coefficient."
    )
    with st.expander("What do the evaluation metrics mean?"):
        for name, explanation in METRIC_HELP.items():
            st.markdown(f"**{name}:** {explanation}")
        st.markdown(
            "Precision, recall and F1 score use **ready biodegradable (RB)** as the positive class. Accuracy, AUC, precision, recall and F1 range from 0 to 1. Higher values indicate better performance for the measured criterion; consider all six metrics together."
        )
        st.markdown(
            "**N/A** means **not applicable**: AUC cannot be calculated when the test file contains only one actual class. AUC uses continuous ready biodegradability scores, rather than predicted class labels."
        )
    if frame[TARGET].nunique() < 2:
        st.warning(
            "This file contains one class only. Area under the receiver operating characteristic curve (AUC) is undefined and shown as N/A (not applicable); the other metrics are still displayed."
        )


def render_comparison(comparison: pd.DataFrame, metric_names: list[str]) -> None:
    """Show all five models with the same six metrics and a downloadable table."""
    with st.container(key="panel_comparison"):
        st.subheader("Compare all five models")
        st.caption(
            "All models use the same active test data. The best value in each metric is highlighted; these results do not tune the models."
        )
        display_comparison = (
            comparison.set_index("ML Model Name")
            .rename_axis("Classifier")
            .rename(index=MODEL_DISPLAY_NAMES)
        )
        styled_comparison = display_comparison.style.highlight_max(color="#dff0e3", axis=0).format(
            precision=4, na_rep="N/A"
        )
        st.dataframe(
            styled_comparison,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(name, format="%.4f") for name in metric_names
            },
        )
        st.download_button(
            "Download comparison CSV",
            comparison.to_csv(index=False).encode(),
            "model_comparison.csv",
            "text/csv",
        )


def render_diagnostics(detail: dict[str, Any]) -> None:
    """Display the selected model's confusion matrix and class-level report."""
    with st.container(key="diagnostics"):
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            with st.container(key="panel_confusion"):
                st.subheader("Confusion matrix")
                st.caption(
                    "Each cell counts records. The diagonal contains correct predictions; the other cells contain mistakes."
                )
                fig = confusion_figure(detail["confusion_matrix"])
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        with right:
            with st.container(key="panel_report"):
                st.subheader("Classification report")
                report = pd.DataFrame(
                    {
                        key: value
                        for key, value in detail["classification_report"].items()
                        if isinstance(value, dict)
                    }
                ).T
                report = report.rename(
                    index={
                        "NRB": "Not ready biodegradable (NRB)",
                        "RB": "Ready biodegradable (RB)",
                        "macro avg": "Unweighted average",
                        "weighted avg": "Weighted average",
                    }
                )
                st.dataframe(
                    report,
                    width="stretch",
                    column_config={
                        name: st.column_config.NumberColumn(
                            {"f1-score": "F1 score", "support": "Examples"}.get(name, name.title()),
                            format="%d" if name == "support" else "%.4f",
                        )
                        for name in report.columns
                    },
                )
                st.caption(
                    "Examples is the number of actual records in each class. The unweighted average gives both classes equal weight; the weighted average uses each class's number of examples."
                )


def render_roc_curves(models: dict[str, Pipeline], frame: pd.DataFrame) -> None:
    """Compare score ranking across thresholds when both actual classes exist."""
    with st.expander("ROC curves across models"):
        st.markdown(
            "**Receiver operating characteristic (ROC)** curves compare true positive rate (recall) with false positive rate as the classification threshold changes. Here, a false positive is a not ready biodegradable chemical predicted as ready biodegradable. The legend gives each model's **area under the curve (AUC)**; 0.5 represents chance ranking and 1.0 represents perfect ranking."
        )
        if frame[TARGET].nunique() == 2:
            # The plotting helper receives evaluated scores, never training data.
            curves = {}
            for name, model in models.items():
                result = evaluate_model(model, frame)
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    frame[TARGET].eq("RB").astype(int), result["scores"]
                )
                curves[name] = (false_positive_rate, true_positive_rate, result["metrics"]["AUC"])
            fig = roc_figure(curves)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "Receiver operating characteristic curves require both ready biodegradable (RB) and not ready biodegradable (NRB) examples."
            )


def render_predictions(frame: pd.DataFrame, detail: dict[str, Any]) -> None:
    """Keep predictions and scores aligned with the original input row order."""
    with st.expander("Inspect predictions and test data"):
        predictions = pd.DataFrame(
            {
                "Row": np.arange(1, len(frame) + 1),
                "Actual class": frame[TARGET].to_numpy(),
                "Predicted class": detail["predictions"],
                "Ready biodegradable score": detail["scores"],
            }
        )
        st.dataframe(predictions, hide_index=True, width="stretch")
        st.caption(
            "Ready biodegradable score is a decision value for the Support Vector Machine (SVM) and a probability for the other models. NRB means not ready biodegradable; RB means ready biodegradable. Scores from different model types are not directly comparable."
        )
        st.download_button(
            "Download predictions",
            predictions.to_csv(index=False).encode(),
            "predictions.csv",
            "text/csv",
        )
        st.dataframe(frame.head(20), hide_index=True, width="stretch")


def render_dataset_method(root: Path) -> None:
    """Document data provenance, preprocessing and the required upload schema."""
    with st.container(key="panel_method"):
        st.subheader("A shared task, five approaches")
        st.markdown(
            "The **QSAR biodegradation** dataset comes from the **University of California, Irvine (UCI) Machine Learning Repository**. **QSAR** stands for **Quantitative Structure–Activity Relationship**: relating molecular structure to a measured property. The dataset contains 1,055 chemical records, 41 numerical molecular properties and an experimentally determined class. The task is to predict **ready biodegradable (RB)** or **not ready biodegradable (NRB)**."
        )
        st.markdown(
            "[Dataset, attribution, license and descriptor definitions](https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation) · [Permanent dataset reference](https://doi.org/10.24432/C5H60M)"
        )
        st.markdown(
            "Three duplicate rows are removed before an **80% training / 20% test split**. The split is stratified, so the class proportions remain similar, and uses random seed 42 so it can be reproduced. The 211 test records are held out from training. Standardization is learned from training data only and saved with each applicable classifier. The Decision Tree uses the original property scales."
        )
        st.markdown(
            "The included comma-separated values (CSV) file is the test set used for the project documentation's results. Uploaded results describe the selected file; use records not used in training to assess performance on new data. Model settings are fixed before evaluation. The documented overall winner describes this test set."
        )
        manifest_path = root / "results" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            with st.expander("Reproducibility record"):
                st.json(manifest)
    with st.container(key="panel_schema"):
        st.subheader("Required CSV columns")
        st.dataframe(
            pd.DataFrame({"Feature": FEATURES, "Type": ["Numeric, finite"] * len(FEATURES)}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Include the class column with ready biodegradable (RB) or not ready biodegradable (NRB) labels. All 42 columns may appear in any order. Missing, extra, duplicate or invalid columns are rejected with an explanation."
        )
        st.markdown(
            "**Why do column names look abbreviated?** These are the original scientific molecular-descriptor codes used to train the models. Keep them unchanged. For example, `nO` is the number of oxygen atoms, `nN` is the number of nitrogen atoms and `nX` is the number of halogen atoms. [See the source's definitions for all 41 descriptors](https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation)."
        )


# Application entry point


def main() -> None:
    """Run the app in the same order a faculty evaluator uses the interface."""
    st.set_page_config(
        page_title="Biodegradability | Model Explorer", page_icon="🌿", layout="wide"
    )
    apply_theme()
    render_hero()

    # 1. Select test data and a classifier; reject invalid data before evaluation.
    sample_path = ROOT / "test_data.csv"
    source, uploaded, selected = render_sidebar(sample_path)
    frame = read_selected_test_data(source, uploaded, sample_path)

    # 2. Reuse only our five trusted, already-fitted pipelines from model/.
    try:
        models = saved_models()
    except (OSError, ValueError) as exc:
        st.error(
            f"Saved models could not be loaded: {exc}. Run python train.py from the project folder first."
        )
        st.stop()

    # 3. Every classifier receives the same test rows; the target is never input.
    with st.spinner("Evaluating saved classifiers..."):
        comparison = evaluate_models(models, frame)
        detail = evaluate_model(models[selected], frame)

    # 4. Display the data context, model comparison and detailed diagnostics.
    render_test_data_context(frame, source)
    evaluation_tab, method_tab = st.tabs(["Model evaluation", "Dataset & method"])
    with evaluation_tab:
        with st.container(key="selected_model"):
            st.subheader(MODEL_DISPLAY_NAMES[selected])
            st.caption(MODEL_NOTES[selected])
        render_metrics(detail["metrics"])
        render_metric_guide(frame)
        render_comparison(comparison, list(detail["metrics"]))
        render_diagnostics(detail)
        render_roc_curves(models, frame)
        render_predictions(frame, detail)
    with method_tab:
        render_dataset_method(ROOT)


if __name__ == "__main__":
    main()

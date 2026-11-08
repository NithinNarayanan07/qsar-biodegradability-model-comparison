"""Streamlit evaluation UI. Training is a separate, offline step in train.py."""

from pathlib import Path
import json
import os

# Keep plotting caches inside this project when running in a restricted lab.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from ml_core import (
    CLASS_LABELS, FEATURES, MODEL_FILES, TARGET,
    evaluate_model, evaluate_models, load_models, read_test_csv,
)


MODEL_NOTES = {
    "Logistic Regression": "A linear baseline with standardized descriptors. Its RB probability is used for AUC.",
    "Decision Tree": "A tree learns feature thresholds and interactions. Its RB probability is used for AUC.",
    "KNN": "Nearest-neighbour voting uses standardized descriptors so large-scale features do not dominate distance.",
    "Naive Bayes": "Gaussian Naive Bayes models class-specific feature distributions and assumes conditional independence.",
    "Support Vector Machine": "An RBF kernel models nonlinear boundaries. AUC uses its continuous RB decision score, not a probability.",
}


@st.cache_resource
def saved_models():
    """Load only the five artifacts shipped with this project, never user files."""
    return load_models(ROOT / "model")


def confusion_figure(matrix):
    """Use a fixed class order so a one-class upload still gets a 2x2 chart."""
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 3.3), layout="constrained")
    ax.imshow(values, cmap="Greens", vmin=0)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_LABELS,
           yticklabels=CLASS_LABELS, xlabel="Predicted class", ylabel="Actual class")
    for (row, col), value in np.ndenumerate(values):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=19,
                color="white" if value > values.max() / 2 else "#182e26")
    ax.spines[:].set_visible(False)
    return fig


def main():
    st.set_page_config(page_title="Biodegradability | Model Explorer", page_icon="🌿", layout="wide")
    st.caption("MACHINE LEARNING · ASSIGNMENT 2")
    st.title("Biodegradability, compared.")
    st.write("Explore how five classifiers distinguish ready biodegradable chemicals from those that are not ready biodegradable.")

    with st.sidebar:
        st.header("Evaluation workspace")
        source = st.radio("Test data source", ["Included test set", "Upload test CSV"])
        sample_path = ROOT / "test_data.csv"
        if sample_path.exists():
            st.download_button("Download test CSV", sample_path.read_bytes(), "test_data.csv", "text/csv", width="stretch")
        st.caption("CSV format: 41 numeric descriptors plus a class column containing NRB or RB. Maximum file size: 5 MB.")
        uploaded = st.file_uploader("Upload labeled test data", type=["csv"], disabled=source == "Included test set")
        selected = st.selectbox("Classifier", list(MODEL_FILES))
        st.divider()
        st.markdown("**Class key**\n\n**RB** · Ready biodegradable\n\n**NRB** · Not ready biodegradable")
        st.caption("Models are trained offline. Uploads are used for evaluation only and are not saved to disk.")

    if source == "Upload test CSV" and uploaded is None:
        st.info("Upload a labeled test CSV to compare all five classifiers. Download the included file for the required column format.")
        st.stop()
    try:
        frame = read_test_csv(uploaded.getvalue() if uploaded is not None and source == "Upload test CSV" else sample_path)
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(f"Cannot evaluate this CSV: {exc}")
        st.stop()
    try:
        models = saved_models()
    except (OSError, ValueError) as exc:
        st.error(f"Saved models could not be loaded: {exc}. Run python train.py from the project folder first.")
        st.stop()

    with st.spinner("Evaluating saved classifiers..."):
        comparison = evaluate_models(models, frame)
        detail = evaluate_model(models[selected], frame)

    counts = frame[TARGET].value_counts()
    st.caption(f"{'Included held-out test set' if source == 'Included test set' else 'Uploaded test data'} · {len(frame):,} rows · {len(FEATURES)} features · {counts.get('NRB', 0)} NRB / {counts.get('RB', 0)} RB")
    evaluation_tab, method_tab = st.tabs(["Model evaluation", "Dataset & method"])
    with evaluation_tab:
        st.subheader(selected)
        st.caption(MODEL_NOTES[selected])
        for column, (name, value) in zip(st.columns(6), detail["metrics"].items()):
            column.metric(name, "N/A" if value is None else f"{value:.4f}")
        st.caption("Precision, recall and F1 treat RB as the positive class. AUC uses continuous model scores; MCC accounts for all four confusion-matrix counts.")
        if frame[TARGET].nunique() < 2:
            st.warning("This file contains one class only. ROC AUC is undefined; the other metrics are still displayed.")

        st.subheader("Compare all five models")
        st.caption("Every row is evaluated on the same active test data. Higher values are better for all six metrics; these results do not tune the models.")
        st.dataframe(comparison.set_index("ML Model Name"), width="stretch",
                     column_config={name: st.column_config.NumberColumn(name, format="%.4f") for name in detail["metrics"]})
        st.download_button("Download comparison CSV", comparison.to_csv(index=False).encode(), "model_comparison.csv", "text/csv")

        left, right = st.columns([1, 1.3], gap="large")
        with left:
            st.subheader("Confusion matrix")
            fig = confusion_figure(detail["confusion_matrix"])
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right:
            st.subheader("Classification report")
            report = pd.DataFrame({key: value for key, value in detail["classification_report"].items() if isinstance(value, dict)}).T
            st.dataframe(report, width="stretch", column_config={
                name: st.column_config.NumberColumn(name.title(), format="%d" if name == "support" else "%.4f")
                for name in report.columns
            })
            st.caption("Rows are actual classes; columns in the matrix are predicted classes. Support is the number of true examples in each report row.")

        with st.expander("ROC curves across models"):
            if frame[TARGET].nunique() == 2:
                fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
                for name, model in models.items():
                    result = evaluate_model(model, frame)
                    fpr, tpr, _ = roc_curve(frame[TARGET].eq("RB").astype(int), result["scores"])
                    ax.plot(fpr, tpr, label=f"{name} ({result['metrics']['AUC']:.3f})")
                ax.plot([0, 1], [0, 1], "--", color="#93a49a", label="Chance")
                ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
                ax.legend(fontsize=8, loc="lower right")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ROC curves require both RB and NRB examples.")

        with st.expander("Inspect predictions and test data"):
            predictions = pd.DataFrame({"Row": np.arange(1, len(frame) + 1), "Actual class": frame[TARGET].to_numpy(),
                                        "Predicted class": detail["predictions"], "RB score": detail["scores"]})
            st.dataframe(predictions, hide_index=True, width="stretch")
            st.caption("RB score is a decision value for SVM and a probability for the other models. Scores from different model types are not directly comparable.")
            st.download_button("Download predictions", predictions.to_csv(index=False).encode(), "predictions.csv", "text/csv")
            st.dataframe(frame.head(20), hide_index=True, width="stretch")

    with method_tab:
        st.subheader("A shared task, five approaches")
        st.markdown("The **UCI QSAR biodegradation** dataset contains 1,055 chemical records, 41 numerical molecular descriptors and an experimental class. The task is to predict **RB** or **NRB** from those descriptors.")
        st.markdown("[Dataset, attribution and license](https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation) · [Dataset DOI](https://doi.org/10.24432/C5H60M)")
        st.markdown("An 80/20 stratified split with random seed 42 creates a reproducible holdout. Duplicate feature vectors are handled before splitting. Scaling is fitted on the training partition and saved with each applicable classifier. The decision tree uses the original feature scale.")
        st.markdown("The included CSV is the holdout used for the README results. Uploaded results describe that uploaded file; use genuinely unseen data when assessing generalization. Model settings are fixed before evaluation. The overall winner in the README describes this holdout, rather than a universal best classifier.")
        manifest_path = ROOT / "results" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            with st.expander("Reproducibility record"):
                st.json(manifest)
        st.subheader("Required CSV columns")
        st.dataframe(pd.DataFrame({"Feature": FEATURES, "Type": ["Numeric, finite"] * len(FEATURES)}), hide_index=True, width="stretch")
        st.caption("Add class as the final column, using RB or NRB. Column order is flexible; missing, extra, duplicate or invalid columns are rejected with an explanation.")


if __name__ == "__main__":
    main()

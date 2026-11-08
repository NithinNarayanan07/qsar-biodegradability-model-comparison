"""Theme, metric cards and chart builders for the Streamlit interface.

This module presents existing values; it never loads files, evaluates a model
or starts training. The app passes precomputed metrics and chart coordinates.
Colors and display names use the stable model keys defined in ml_core.py.
"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np
import streamlit as st

# Keep a classifier's color consistent between its name and the ROC chart.
MODEL_COLORS = {
    "Logistic Regression": "#167447",
    "Decision Tree": "#178e91",
    "KNN": "#4978bc",
    "Naive Bayes": "#b47b23",
    "Support Vector Machine": "#8060ae",
}

# Expand names for faculty without changing saved filenames or result identifiers.
MODEL_DISPLAY_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "KNN": "K-Nearest Neighbours (KNN)",
    "Naive Bayes": "Gaussian Naive Bayes",
    "Support Vector Machine": "Support Vector Machine (SVM)",
}

METRIC_HELP = {
    "Accuracy": "The fraction of all test examples classified correctly.",
    "AUC": "Area under the receiver operating characteristic (ROC) curve. Measures how well scores rank ready biodegradable chemicals above not ready biodegradable chemicals. Requires both classes.",
    "Precision": "Among chemicals predicted ready biodegradable, the fraction that are actually ready biodegradable.",
    "Recall": "Among actual ready biodegradable chemicals, the fraction identified correctly.",
    "F1": "F1 score: the harmonic mean of precision and recall for the ready biodegradable class.",
    "MCC": "Matthews correlation coefficient. Measures agreement between actual and predicted classes using all four confusion-matrix cells. Ranges from −1 to +1; +1 means perfect agreement.",
}


def apply_theme() -> None:
    """Load local CSS; no remote fonts, scripts or image assets are required."""
    st.html(Path(__file__).with_name("styles.css"))


def render_hero() -> None:
    """Introduce the chemical classification task and expand the QSAR name."""
    with st.container(key="hero"):
        st.html('<div class="hero-eyebrow"><span></span> QSAR MODEL EXPLORER</div>')
        st.title("Biodegradability, compared.")
        st.write(
            "Explore how five classifiers distinguish ready biodegradable chemicals from those that are not ready biodegradable."
        )
        st.caption(
            "Quantitative Structure–Activity Relationship (QSAR): linking molecular structure to a measured property."
        )
        st.html("""<div class="hero-facts">
            <span><b>05</b> Classifiers</span>
            <span><b>41</b> Molecular descriptors</span>
            <span><b>06</b> Evaluation metrics</span>
        </div>""")


def render_sidebar_brand() -> None:
    """Render the same local brand assets on a desktop or hosted app."""
    st.html("""<div class="sidebar-brand">
        <div class="brand-symbol" aria-hidden="true"></div>
        <div><strong>QSAR</strong><span>Biodegradability lab</span></div>
    </div>""")
    st.caption("MACHINE LEARNING · ASSIGNMENT 02")


def sidebar_step(number: str, label: str) -> None:
    """Render only fixed application copy, never uploaded content."""
    st.html(f'<div class="sidebar-step"><span>{number}</span><strong>{label}</strong></div>')


def render_metrics(metrics: dict[str, float | None]) -> None:
    """Keep native, accessible metrics in a readable three-by-two arrangement."""
    items = list(metrics.items())
    for start in range(0, len(items), 3):
        for column, (name, value) in zip(st.columns(3), items[start : start + 3]):
            with column.container(key=f"metric_{name.lower()}"):
                st.metric(name, "N/A" if value is None else f"{value:.4f}", help=METRIC_HELP[name])


def style_plot(fig: Figure, ax: Axes) -> None:
    """Apply one readable chart style without changing any plotted values."""
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#536d61", labelsize=9)
    ax.xaxis.label.set_color("#314f42")
    ax.yaxis.label.set_color("#314f42")
    for spine in ax.spines.values():
        spine.set_color("#dce7df")


def confusion_figure(matrix: list[list[int]]) -> Figure:
    """Use a fixed class order so a one-class upload still gets a 2x2 chart."""
    values = np.asarray(matrix)
    fig = Figure(figsize=(5, 3.3), layout="constrained")
    ax = fig.subplots()
    palette = LinearSegmentedColormap.from_list(
        "biodegradability", ["#eff7ef", "#b8dfbf", "#5aa97b", "#18583b"]
    )
    ax.imshow(values, cmap=palette, vmin=0)
    class_names = ["Not ready\nbiodegradable (NRB)", "Ready\nbiodegradable (RB)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    for (row, col), value in np.ndenumerate(values):
        ax.text(
            col,
            row,
            str(value),
            ha="center",
            va="center",
            fontsize=19,
            color="white" if value > values.max() / 2 else "#183c2b",
            fontweight="semibold",
        )
    style_plot(fig, ax)
    ax.spines[:].set_visible(False)
    return fig


def roc_figure(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> Figure:
    """Plot precomputed ROC coordinates; this helper does not evaluate or fit models."""
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    for name, (false_positive_rate, true_positive_rate, auc) in curves.items():
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{MODEL_DISPLAY_NAMES[name]} (AUC {auc:.3f})",
            color=MODEL_COLORS[name],
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "--", color="#9aaba0", label="Chance", linewidth=1.1)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.14, color="#8fae98")
    ax.legend(fontsize=8, loc="lower right", facecolor="#ffffff", edgecolor="#dce7df")
    style_plot(fig, ax)
    return fig

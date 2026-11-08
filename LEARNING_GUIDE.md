# Understand and reproduce the solution

Use this guide alongside the code and your own run. The assignment permits AI as learning support; review the decisions, reproduce the results, and write observations you can explain before submitting.

## Follow one example through the project

1. `fetch_data.py` obtains the original UCI dataset and records its provenance.
2. `train.py` removes exact duplicate records, separates training and test rows, fits five pipelines and saves them under `model/`.
3. `ml_core.py` validates test data, loads the saved pipelines, predicts labels and computes metrics.
4. `app.py` displays those evaluations in Streamlit. It does not train a model.
5. `test_data.csv` contains the labeled holdout used for the comparison in the README.

The target is `class`. **RB** means ready biodegradable and is the positive class; **NRB** means not ready biodegradable and is the negative class. A numerical descriptor represents a property of a molecule, not its target label.

## Why the split comes before scaling

A model should be evaluated on examples it has not learned from. We use a reproducible 80/20 stratified split with random seed 42. Stratification keeps approximately the same class proportions in each partition.

Three exact duplicate records are removed before splitting. Otherwise, an identical feature vector could appear in both partitions and make the test result overly optimistic. The original downloaded file remains available for inspection.

`StandardScaler` estimates each feature's mean and standard deviation. These estimates must come from training rows only. A scikit-learn `Pipeline` stores the fitted scaler with its classifier, so the same transformation is applied when the saved model sees test data. Fitting a scaler to the whole dataset would leak information from the test distribution.

The decision tree does not need scaling: a threshold such as `feature < 4` can be transformed without changing the underlying ordering. Distance-based models, particularly KNN and RBF SVM, are sensitive to feature scales.

## What each classifier learns

| Classifier | Mechanism | Question to consider |
| --- | --- | --- |
| Logistic Regression | A weighted linear combination of standardized descriptors produces a class probability. | What patterns could a linear boundary miss? |
| Decision Tree | Repeated feature thresholds divide the training data into regions. | How does limiting tree depth reduce overfitting? |
| KNN | Nearby standardized training examples vote on a new example's class. | How would increasing the number of neighbours change the boundary? |
| Gaussian Naive Bayes | Each feature has a Gaussian distribution within a class; feature likelihoods are combined under conditional independence. | Why might correlated molecular descriptors violate that assumption? |
| Support Vector Machine | An RBF kernel creates a nonlinear decision boundary with a margin. | What roles do `C` and `gamma` play? |

We compare these approaches with fixed settings on one shared holdout. The strongest observed holdout result does not prove that a classifier will be best on every future dataset.

## Read the six metrics

With RB as positive, **TP** is an RB example predicted RB, **FP** is an NRB example predicted RB, **FN** is an RB example predicted NRB, and **TN** is an NRB example predicted NRB.

| Metric | Meaning |
| --- | --- |
| Accuracy | `(TP + TN) / all examples`. Useful overall, but can hide class imbalance. |
| AUC | Area under the ROC curve. Measures how well continuous scores rank RB above NRB as the threshold changes. |
| Precision | `TP / (TP + FP)`. Among RB predictions, how many are truly RB? |
| Recall | `TP / (TP + FN)`. Among actual RB examples, how many are found? |
| F1 | Harmonic mean of precision and recall. Both must be strong for F1 to be high. |
| MCC | A correlation measure using all four confusion-matrix cells. It ranges from -1 to 1; 1 is perfect agreement and 0 indicates no correlation. |

AUC must use continuous scores, not predicted class labels. Logistic Regression, Decision Tree, KNN and Gaussian Naive Bayes provide the RB probability. SVM provides an RB decision score; that score is not a probability, but it can rank examples for ROC AUC. The code looks up the RB class rather than assuming a probability column's position.

If an uploaded file contains only one true class, ROC AUC cannot be calculated and the app displays `N/A`. Undefined precision/recall/F1 divisions use zero; MCC may also be zero for degenerate class patterns. A confusion matrix still uses a fixed `[NRB, RB]` order.

## Inspect the actual saved model

From the project directory, with the virtual environment active:

```bash
python -c 'import joblib; model = joblib.load("model/svm.joblib"); print(model); print(model.classes_); print(model.get_params())'
```

Inspect only artifacts that you generated or trust. Joblib model files use pickle internally; CSV uploads are data, never model files.

To inspect all saved pipelines:

```bash
python -c 'from pathlib import Path; from ml_core import load_models; models = load_models(Path("model")); [print(name, model) for name, model in models.items()]'
```

To recompute the table without training:

```bash
python -c 'from pathlib import Path; from ml_core import read_test_csv, load_models, evaluate_models; print(evaluate_models(load_models(Path("model")), read_test_csv(Path("test_data.csv"))).to_string(index=False))'
```

## Experiments worth doing yourself

- Explain why simply predicting NRB on every row can achieve a seemingly reasonable accuracy, given the class distribution.
- Compare one model's confusion matrix with its precision and recall. Recalculate those two metrics by hand.
- Inspect the scaler's learned statistics and confirm they came only from the training partition.
- If you want to choose hyperparameters, use cross-validation within training data, keep the final holdout untouched, and document your selection rule before evaluation.
- Describe a limitation visible in the actual results rather than assuming a complex classifier must win.

Do not repeatedly change settings to improve the published test table. That turns the test set into a validation set. Any deliberate experimental change should have a clear reason, use training-only validation, and be followed by consistent regeneration of saved artifacts, metrics and README observations.

## What to show in BITS Virtual Lab

Reproduce the environment, run `python train.py`, and launch the app with `python -m streamlit run app.py` inside the lab. Verify the test CSV and classifier selection, then capture genuine execution evidence showing the lab context and results. A local screenshot is useful for development, but it does not demonstrate execution in the lab.

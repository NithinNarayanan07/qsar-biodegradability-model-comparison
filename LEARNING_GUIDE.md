# Understand and reproduce the solution

Use this guide alongside the code and your own run. The assignment permits AI as learning support; review the decisions, reproduce the results, and write observations you can explain before submitting.

## A reading route for faculty and students

Start with the README's problem statement, dataset description, comparison table and observations. For the source review, follow the data from its public origin to the displayed results:

| Read | Module | What to look for |
| --- | --- | --- |
| 1 | `fetch_data.py` | The public download URL, source checksum and provenance record. Existing source data is verified before reuse. |
| 2 | `train.py` | The five model definitions, duplicate handling, shared training/test split, and the fitting and saving of complete pipelines. |
| 3 | `ml_core.py` | Required CSV columns, target labels, input checks, loading of fixed model files, predictions and metric calculations. |
| 4 | `app.py` | The Streamlit entry point and named interface sections that coordinate user choices, evaluation and displayed results. |
| 5 | `presentation.py` | The theme, brand, metric-card explanations and chart builders. Visual styling lives in `styles.css`. |

Comments explain why decisions such as splitting before scaling are necessary. The module boundaries separate scientific calculations from interface presentation, so a change in colors or labels need not change model behavior.

## Follow one record through training and evaluation

The original source has 1,055 records. Removing three duplicate rows leaves 1,052 records, split into **841 training records** and **211 separate test records**. Each contains 41 numerical molecular properties and a known class label. `test_data.csv` contains the 211-row holdout used for the README comparison.

The target is `class`. **RB** means ready biodegradable and is the positive class; **NRB** means not ready biodegradable and is the negative class. A numerical descriptor represents a property of a molecule, not its target label.

During **offline training**, `train.py` reads the verified dataset through `fetch_data.py`, creates the split and fits five pipelines using only training rows. A pipeline saves its fitted preprocessing and classifier together in one `.joblib` file. The decision-tree pipeline needs only its classifier. The script also writes the test CSV, metric comparison, detailed evaluations and reproducibility manifest.

During **app evaluation**, `app.py` asks `ml_core.py` to validate the selected labeled CSV and load those saved pipelines. Each model receives the 41 input properties, produces predictions and a continuous score, and its predictions are compared with the known `class` values. `presentation.py` supplies the charts and styled metric cards displayed by the app. Known labels are answers used for evaluation; they are not model inputs or predictions.

Changing the classifier, uploading a file or refreshing the page repeats evaluation with saved models. It does **not** retrain them. Only explicitly running `python train.py` starts a new training run and rewrites the generated artifacts.

## Reproduce or evaluate

With the environment installed as described in the README, choose the operation you need:

| Command | Effect |
| --- | --- |
| `python -m streamlit run app.py` | Evaluate the shipped saved models on the provided test set or a compatible uploaded file. |
| `python train.py` | Reproduce offline training and regenerate `model/`, `test_data.csv` and `results/`; then launch the app to inspect the run. |
| `python -m pytest` | Run the verification suite after installing `requirements-dev.txt`. |

An upload must describe the **same chemical biodegradability task**, using the same 41 molecular properties and numerical conventions, plus known RB/NRB labels. An unrelated dataset does not become valid by renaming its columns. Validation checks file structure and values; it cannot prove the origin or scientific appropriateness of the measurements.

## Why the split comes before scaling

A model should be evaluated on examples it has not learned from. We use a reproducible 80/20 stratified split with random seed 42. Stratification keeps approximately the same class proportions in each partition.

Three exact duplicate records are removed before splitting. Otherwise, an identical feature vector could appear in both partitions and make the test result overly optimistic. The original downloaded file remains available for inspection.

`StandardScaler` estimates each feature's mean and standard deviation. These estimates must come from training rows only. A scikit-learn `Pipeline` stores the fitted scaler with its classifier, so the same transformation is applied when the saved model sees test data. Fitting a scaler to the whole dataset would leak information from the test distribution.

The decision tree does not need scaling: a threshold such as `feature < 4` can be transformed without changing the underlying ordering. Distance-based models, particularly K-Nearest Neighbours (KNN) and the radial basis function (RBF) version of Support Vector Machine (SVM), are sensitive to feature scales.

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
| AUC | Area under the receiver operating characteristic (ROC) curve. Measures how well continuous scores rank RB above NRB as the threshold changes. |
| Precision | `TP / (TP + FP)`. Among RB predictions, how many are truly RB? |
| Recall | `TP / (TP + FN)`. Among actual RB examples, how many are found? |
| F1 score | Harmonic mean of precision and recall. Both must be strong for F1 to be high. |
| MCC | Matthews correlation coefficient, using all four confusion-matrix cells. It ranges from -1 to 1; 1 is perfect agreement and 0 indicates no correlation. |

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

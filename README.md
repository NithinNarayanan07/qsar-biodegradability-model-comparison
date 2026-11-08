# QSAR Biodegradability — Five Classifier Comparison

A reproducible Machine Learning Assignment 2 project: train five classifiers on one dataset, save their preprocessing and models, and evaluate labeled test CSV files in Streamlit without retraining.

**Submission status:** the reproducible models and Streamlit app are implemented, and BITS Virtual Lab execution is complete. Repository and app links are below. The final PDF is assembled separately for the student's upload; final portal submission is not claimed here.

## a. Problem statement

Predict whether a chemical is **ready biodegradable (RB)** or **not ready biodegradable (NRB)** from 41 numerical molecular properties, also called descriptors. Compare Logistic Regression, Decision Tree, K-Nearest Neighbours (KNN), Gaussian Naive Bayes and Support Vector Machine (SVM) using the same separate test records and six evaluation metrics.

This is an educational classification experiment. Its results do not establish fitness for environmental or regulatory decisions.

## b. Dataset description

The University of California, Irvine (UCI) Machine Learning Repository publishes this dataset. QSAR means **quantitative structure–activity relationship**: relating a molecule's structure to an observed property or activity.

| Item | Description |
| --- | --- |
| Dataset | UCI QSAR biodegradation, dataset 254 |
| Public source | [UCI dataset page](https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation) |
| Original size | 1,055 records, 41 numeric input features and one target |
| Original class distribution | 699 NRB and 356 RB |
| Target | `class`, with values `NRB` and `RB`; RB is positive |
| Missing values | None in the original dataset |
| Duplicate handling | Three exact duplicate records removed before splitting; 1,052 records remain. Identical feature vectors have no conflicting labels. |
| Split | 80% training / 20% test, stratified by class, `random_state=42` |
| Training partition | 841 records: 558 NRB and 283 RB |
| Test partition | 211 records: 140 NRB and 71 RB |
| Dataset license | [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) |

The dataset exceeds the assignment's minimum of 500 instances and 12 input features. The target is not counted as an input feature. `data/biodeg.csv` preserves the original semicolon-delimited, headerless download. The generated `test_data.csv` is a comma-delimited file with named columns for the app.

The models learn from **841 training records**. The remaining **211 test records** are kept separate from training; this is the held-out test set. Each test row supplies 41 input properties and a known `class` label. The models predict a class from the properties, and those predictions are compared with the known labels to measure performance. The class counts describe known answers, not predictions. All five models receive the same selected test records.

**Attribution:** Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. (2013). *QSAR biodegradation* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5H60M](https://doi.org/10.24432/C5H60M). The underlying study is *Quantitative Structure–Activity Relationship models for ready biodegradability of chemicals*, Journal of Chemical Information and Modeling, 53, 867–878.

The source archive URL, download checksum and data audit are retained in the generated provenance and reproducibility records. No public student solution repository was used as a code source.

## c. GitHub Repository Link

[qsar-biodegradability-model-comparison](https://github.com/NithinNarayanan07/qsar-biodegradability-model-comparison)

**Live Streamlit app:** [QSAR Biodegradability Model Comparison](https://qsar-biodegradability-model-comparison.streamlit.app/)

The deployed app uses branch `codex/ml-assignment-2`, entry point `app.py` and Python 3.12.16. Anonymous app access, all five classifier selections, displayed scores and the test CSV download were verified. The live upload control is enabled; the full CSV upload flow was verified locally and still needs a live browser upload check.

## d. Models used

All five classifiers use the same training and test rows. Standardization is fitted **only on the training data** and saved inside the Logistic Regression, KNN, Gaussian Naive Bayes and SVM pipelines. The Decision Tree uses unscaled descriptors. Model settings are fixed before holdout evaluation; the test set is not used to select hyperparameters.

The SVM uses a **radial basis function (RBF)** kernel to learn a nonlinear decision boundary.

| Model | Main fixed settings |
| --- | --- |
| Logistic Regression | `C=1.0`, `solver="lbfgs"`, `max_iter=3000` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=4`, Gini impurity |
| KNN | 9 neighbours, uniform voting, Euclidean distance |
| Naive Bayes | GaussianNB, `var_smoothing=1e-9` |
| Support Vector Machine | RBF kernel, `C=1.0`, `gamma="scale"` |

### Evaluation metrics

<!-- METRICS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8626 | 0.9114 | 0.8088 | 0.7746 | 0.7914 | 0.6893 |
| Decision Tree | 0.8294 | 0.8494 | 0.7397 | 0.7606 | 0.7500 | 0.6207 |
| KNN | 0.8531 | 0.9102 | 0.7500 | 0.8451 | 0.7947 | 0.6839 |
| Naive Bayes | 0.6540 | 0.8020 | 0.4926 | 0.9437 | 0.6473 | 0.4450 |
| Support Vector Machine | 0.8531 | 0.9211 | 0.7941 | 0.7606 | 0.7770 | 0.6679 |
<!-- METRICS_TABLE_END -->

These are measured results from the 211-row held-out `test_data.csv`. Full-precision metrics are stored in `results/metrics.csv`; confusion matrices and classification reports are in `results/evaluation.json`.

**AUC** is the area under the **receiver operating characteristic (ROC)** curve; it measures how well model scores rank RB examples above NRB examples across decision thresholds. **F1 score** balances precision and recall using their harmonic mean. **MCC** means **Matthews correlation coefficient**, an overall agreement measure using correct and incorrect predictions for both classes, ranging from -1 to 1.

Precision, recall and F1 use binary averaging with **RB as positive**. Accuracy and MCC use predicted labels. AUC uses RB probabilities for Logistic Regression, Decision Tree, KNN and Gaussian Naive Bayes, and the continuous RB decision score for SVM. Scores are not rounded before metrics are computed; the displayed values are rounded to four decimal places.

When a CSV contains only one actual class, AUC is undefined and the app displays `N/A`. Undefined precision/recall/F1 divisions return zero. MCC can be zero for degenerate class patterns. Confusion matrices always use `[NRB, RB]` for both axes.

### Observations about model performance

<!-- OBSERVATIONS_START -->
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Highest accuracy (0.8626), precision (0.8088) and MCC (0.6893). It correctly classifies 127 of 140 NRB and 55 of 71 RB examples. Its strong result suggests a linear baseline captures useful signal in this feature representation, although 16 RB examples are missed. |
| Decision Tree | Accuracy is 0.8294 and MCC is 0.6207, below Logistic Regression, KNN and SVM. It makes 19 false-positive RB predictions and misses 17 actual RB examples. The bounded tree provides a nonlinear comparison, but these results do not establish that its settings are optimal. |
| KNN | Highest F1 (0.7947) and second-highest recall (0.8451). It finds 60 of 71 RB examples, five more than Logistic Regression, at the cost of 20 false positives rather than 13. Accuracy is 0.8531 and MCC (0.6839) is close to Logistic Regression. |
| Naive Bayes | Highest recall (0.9437), finding 67 of 71 RB examples, but it produces 69 false positives. Precision is only 0.4926 and accuracy is 0.6540, slightly below always predicting NRB on this test set (140/211 = 0.6635). Correlated descriptors may conflict with its independence assumption; that is a possible explanation, not a measured causal conclusion. |
| Support Vector Machine | Highest AUC (0.9211), indicating the strongest RB-versus-NRB ranking on this holdout. Accuracy is 0.8531, with 14 false positives and 17 false negatives at the model's default decision threshold. Better ranking does not automatically imply the best fixed-threshold F1 or accuracy. |
| **Overall winner for this dataset?** | **Logistic Regression is the overall holdout winner when using MCC as the primary comparison criterion, supported by its leading accuracy and precision.** KNN is preferable on the observed F1 tradeoff, and SVM leads AUC. Small differences on one test split should not be treated as statistically conclusive. |
<!-- OBSERVATIONS_END -->

The class imbalance makes accuracy alone incomplete. Precision, recall, F1 and MCC help interpret errors on RB, the smaller class. Results describe one finite holdout, and no uncertainty interval or external chemical validation is claimed. Molecular similarity beyond exact duplicates could still make examples dependent; this split is not a scaffold-based evaluation.

## Project files

```text
ml_assignment2_qsar/
├── app.py                     # Streamlit entry point, named UI sections and coordination
├── presentation.py            # Theme, brand, metric cards and chart builders
├── styles.css                 # Green application theme and responsive panels
├── ml_core.py                 # Schema, CSV validation, saved-model evaluation
├── train.py                   # Reproducible training and artifact generation
├── fetch_data.py              # Download and provenance for the UCI data
├── requirements.txt           # Pinned runtime dependencies
├── requirements-dev.txt       # Runtime dependencies plus pytest
├── .python-version            # Python 3.12
├── .streamlit/config.toml     # Theme and 5 MB upload limit
├── test_data.csv              # Labeled holdout used for the results above
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── data/                      # Original dataset and provenance
├── results/
│   ├── metrics.csv
│   ├── evaluation.json
│   └── manifest.json
├── tests/                     # Validation, metric and app checks
├── LEARNING_GUIDE.md
└── README.md
```

The `.venv/`, cache directories and secrets are excluded from Git. The saved models and test CSV must be included when publishing.

### Reading the code

For a faculty review, first read sections a–d for the task, dataset and measured results. Then follow the source in this order:

| Order | Module | Responsibility |
| --- | --- | --- |
| 1 | `fetch_data.py` | Obtain the original public dataset, verify its checksum and record its source and attribution. A verified local copy supports later offline runs. |
| 2 | `train.py` | Define the five classifiers, remove duplicate rows, split the data, fit training-only preprocessing, and save complete pipelines and holdout results. |
| 3 | `ml_core.py` | Define the feature and label schema, validate test CSV files, load the saved pipelines, predict classes and calculate all six metrics. |
| 4 | `app.py` | Start Streamlit and coordinate named interface sections: data/model selection, test-data explanation, evaluation, comparisons and diagnostics. |
| 5 | `presentation.py` | Apply the local theme and render the brand, metric cards and charts; `styles.css` contains the visual rules. |

Training and evaluation have separate entry points. **`python train.py`** fits and writes the model artifacts. **`python -m streamlit run app.py`** loads those saved artifacts and evaluates the selected test records. Each saved pipeline contains the fitted classifier and any required scaler, so uploads receive the same preprocessing learned from the training data. Changing a model, uploading a CSV or refreshing the interface does not call model fitting.

The `tests/` directory checks CSV validation, metric calculations, the saved split and preprocessing, and application behavior. `LEARNING_GUIDE.md` explains how to trace this flow and inspect the saved models yourself.

## Run locally

Use **Python 3.12**. The local environment was created with Python 3.12.14; Streamlit Community Cloud runs Python 3.12.16. Keep the same Python minor version and pinned scikit-learn version on the lab machine and in Streamlit Community Cloud.

On macOS or Linux, from the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is unavailable, invoke the environment's interpreter directly: `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.

The shipped models support evaluation immediately. To reproduce training and regenerate the test data, saved pipelines and results:

```bash
python train.py
```

This reads the verified source file under `data/`, writes five fitted pipelines under `model/`, and regenerates `test_data.csv` and `results/`. Start the app afterward to inspect that run. To evaluate without changing the trained artifacts, skip this command and launch the app directly.

Inspect `train.py` and `results/manifest.json` for the recorded settings and split. Retraining is a deliberate offline action, not an upload action. Recheck the README table if you intentionally change the dataset or model settings.

For the verification suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Use the Streamlit app

1. Start with **Included test set**. The app evaluates every saved classifier on the shipped holdout and displays a comparison table.
2. Choose a **Classifier** to see its six metrics, confusion matrix and classification report.
3. Download the supplied test CSV to inspect the accepted format.
4. Choose **Upload test CSV**, then upload labeled data for this same chemical biodegradability task, using the required molecular-property columns and known class labels.
5. Inspect the updated comparison and per-model diagnostics. Download the comparison or predictions if useful.

The app does not retrain or save uploads to disk. Use unseen data to assess generalization. Scores on training rows or altered labels do not represent a valid test result.

The viewer hides Streamlit's developer menu. Results update automatically when the model or test data changes; a refresh evaluates the saved models without retraining them.

### CSV contract

Uploads support **this chemical biodegradability task only**. The columns must represent the same 41 molecular properties, with the same scientific meanings and numerical conventions as the training dataset, and `class` must give the known biodegradability category. An unrelated dataset is not suitable merely because its columns have been renamed to match. File checks validate structure, numeric values and labels; they cannot establish data provenance or scientific appropriateness. Download the provided test file as the format example.

The file must be UTF-8, comma-separated, contain a header and at least one labeled data row, and be no larger than 5 MB. Include exactly these 41 numeric descriptors plus `class`:

```text
SpMax_L,J_Dz(e),nHM,F01[N-N],F04[C-N],NssssC,nCb-,C%,nCp,nO,F03[C-N],SdssC,HyWi_B(m),LOC,SM6_L,F03[C-O],Me,Mi,nN-N,nArNO2,nCRX3,SpPosA_B(p),nCIR,B01[C-Br],B03[C-Cl],N-073,SpMax_A,Psi_i_1d,B04[C-Br],SdO,TI2_L,nCrt,C-026,F02[C-N],nHDon,SpMax_B(m),Psi_i_A,nN,SM6_B(m),nArCOOR,nX,class
```

Column order is flexible. Surrounding whitespace in headers and class labels is trimmed. Feature cells must be numeric and finite; target cells must be `NRB` or `RB`. Missing or extra columns, duplicate headers, missing cells, malformed records, nonnumeric features, infinity, and unknown labels produce a validation message instead of evaluation. A file with one true class is accepted, with the AUC limitation described above. Do not add an exported dataframe index column.

Only project-generated saved models are loaded. Uploaded model files are not supported.

## Deployment reference

The project is already published and deployed at the links in section c. To reproduce this deployment:

1. Use GitHub repository **`NithinNarayanan07/qsar-biodegradability-model-comparison`**. Keep the source, dependency files, `test_data.csv`, `model/`, `data/`, results and documentation in the repository; exclude `.venv/` and secrets.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and create an app from that repository.
3. Select branch **`codex/ml-assignment-2`**, entry point **`app.py`**, and **Python 3.12** in advanced settings.
4. After deployment, verify the included test set, all five classifier choices, a real CSV upload and downloads. The test CSV download should match the repository's `test_data.csv`.
5. Verify both links from the perspective of an evaluator with the intended access. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Dependency versions are pinned because saved scikit-learn artifacts should be loaded with the version that produced them. `pyarrow==24.0.0` matches the working Community Cloud environment and avoids an Arrow 26.0.1 compatibility issue detected during deployment. An app loading successfully on the developer machine does not replace deployment verification.

## BITS Virtual Lab and final submission

The project was executed inside **BITS Virtual Lab**. Source commit `96c69bb` was downloaded into the lab because Git was unavailable. A Python **3.12.7** virtual environment was created with the pinned dependencies. All five classifiers were trained there, and their evaluation metrics matched the README comparison to four decimal places. The Streamlit app was then launched inside the lab and opened in Firefox at `localhost:8501`.

A single user-captured execution screenshot, with private connection details concealed, is provided in the final submission PDF. The screenshot and student identification are kept in the private submission artifacts rather than the public repository. Local and Community Cloud previews are separate from this lab evidence. See `screenshots/README.md` for the preview descriptions.

The final PDF is assembled separately in the required order:

1. Clickable GitHub repository link.
2. Clickable live Streamlit Community Cloud app link.
3. BITS Virtual Lab execution screenshot.
4. The complete final README content, including the measured comparison table and observations.

The student must upload and submit the final PDF in the assignment portal, rather than leave a draft. Before doing so, verify the links and app again and confirm all required files and the BITS execution image are present. The assignment brief states a deadline of **8 November 2026 at 23:59**.

For an explanation of the code and experiments to perform yourself, read [LEARNING_GUIDE.md](LEARNING_GUIDE.md).

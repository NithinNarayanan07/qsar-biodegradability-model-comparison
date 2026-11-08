# QSAR Biodegradability — Five Classifier Comparison

A reproducible Machine Learning Assignment 2 project: train five classifiers on one dataset, save their preprocessing and models, and evaluate labeled test CSV files in Streamlit without retraining.

**Current stage:** local implementation verified. Streamlit Community Cloud deployment, BITS Virtual Lab execution evidence and final PDF submission remain to be completed.

## a. Problem statement

Predict whether a chemical is **ready biodegradable (RB)** or **not ready biodegradable (NRB)** from 41 numerical molecular descriptors. Compare Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine using the same held-out examples and six evaluation metrics.

This is an educational classification experiment. Its results do not establish fitness for environmental or regulatory decisions.

## b. Dataset description

| Item | Description |
| --- | --- |
| Dataset | UCI QSAR biodegradation, dataset 254 |
| Public source | [UCI dataset page](https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation) |
| Original size | 1,055 records, 41 numeric input features and one target |
| Original class distribution | 699 NRB and 356 RB |
| Target | `class`, with values `NRB` and `RB`; RB is positive |
| Missing values | None in the original dataset |
| Duplicate handling | Three exact duplicate records removed before splitting; 1,052 unique records remain. Identical feature vectors have no conflicting labels. |
| Split | 80% training / 20% test, stratified by class, `random_state=42` |
| Training partition | 841 records: 558 NRB and 283 RB |
| Test partition | 211 records: 140 NRB and 71 RB |
| Dataset license | [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) |

The dataset exceeds the assignment's minimum of 500 instances and 12 input features. The target is not counted as an input feature. `data/biodeg.csv` preserves the original semicolon-delimited, headerless download. The generated `test_data.csv` is a comma-delimited file with named columns for the app.

**Attribution:** Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. (2013). *QSAR biodegradation* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5H60M](https://doi.org/10.24432/C5H60M). The underlying study is *Quantitative Structure–Activity Relationship models for ready biodegradability of chemicals*, Journal of Chemical Information and Modeling, 53, 867–878.

The source archive URL, download checksum and data audit are retained in the generated provenance and reproducibility records. No public student solution repository was used as a code source.

## c. GitHub Repository Link

[qsar-biodegradability-model-comparison](https://github.com/NithinNarayanan07/qsar-biodegradability-model-comparison)

**Live Streamlit app:** pending deployment to Streamlit Community Cloud. Replace this status with the actual live app URL after testing the deployed application.

## d. Models used

All five classifiers use the same training and test rows. Standardization is fitted **only on the training data** and saved inside the Logistic Regression, KNN, Gaussian Naive Bayes and SVM pipelines. The Decision Tree uses unscaled descriptors. Model settings are fixed before holdout evaluation; the test set is not used to select hyperparameters.

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
├── app.py                     # Streamlit evaluation interface
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

## Run locally

Use **Python 3.12**. The local environment was created with Python 3.12.14. Keep the same Python minor version and pinned scikit-learn version on the lab machine and in Streamlit Community Cloud.

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
4. Choose **Upload test CSV**, then upload a labeled file with the same input columns and target.
5. Inspect the updated comparison and per-model diagnostics. Download the comparison or predictions if useful.

The app does not retrain or save uploads to disk. Use unseen data to assess generalization. Scores on training rows or altered labels do not represent a valid test result.

### CSV contract

The file must be UTF-8, comma-separated, contain a header and at least one labeled data row, and be no larger than 5 MB. Include exactly these 41 numeric descriptors plus `class`:

```text
SpMax_L,J_Dz(e),nHM,F01[N-N],F04[C-N],NssssC,nCb-,C%,nCp,nO,F03[C-N],SdssC,HyWi_B(m),LOC,SM6_L,F03[C-O],Me,Mi,nN-N,nArNO2,nCRX3,SpPosA_B(p),nCIR,B01[C-Br],B03[C-Cl],N-073,SpMax_A,Psi_i_1d,B04[C-Br],SdO,TI2_L,nCrt,C-026,F02[C-N],nHDon,SpMax_B(m),Psi_i_A,nN,SM6_B(m),nArCOOR,nX,class
```

Column order is flexible. Surrounding whitespace in headers and class labels is trimmed. Feature cells must be numeric and finite; target cells must be `NRB` or `RB`. Missing or extra columns, duplicate headers, missing cells, malformed records, nonnumeric features, infinity, and unknown labels produce a validation message instead of evaluation. A file with one true class is accepted, with the AUC limitation described above. Do not add an exported dataframe index column.

Only project-generated saved models are loaded. Uploaded model files are not supported.

## Publish and deploy after local review

1. Create a GitHub repository for this project and publish its source, dependency files, `test_data.csv`, `model/`, `data/`, results and documentation. Do not publish `.venv/` or secrets.
2. Confirm that GitHub displays the required files and README. Replace the pending repository status in section c with the real clickable URL.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and create an app from the repository.
4. Choose the correct branch and **`app.py`** as the entry point. In advanced settings choose **Python 3.12**.
5. Deploy, verify the sample evaluation, all five classifier choices, a real CSV upload and downloads, then record the actual live app URL in section c.
6. Verify both links from the perspective of an evaluator with the intended access. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Dependency versions are pinned because saved scikit-learn artifacts should be loaded with the version that produced them. An app loading successfully on the developer machine does not replace deployment verification.

## BITS Virtual Lab and final submission

Check lab access early. Copy or clone this project inside BITS Virtual Lab, create the Python environment, run the training/evaluation code there, and launch and exercise the app there. Capture **one genuine screenshot of assignment execution with visible lab context**. Keep the result readable. This evidence is still pending; no lab execution or screenshot is claimed by this local project.

Prepare one final PDF in the required order:

1. Clickable GitHub repository link.
2. Clickable live Streamlit Community Cloud app link.
3. BITS Virtual Lab execution screenshot.
4. The complete final README content, including the measured comparison table and observations.

Before submitting, verify the links and app again, confirm all required files are present, replace pending publication statuses, and submit the final PDF rather than leaving a draft. The assignment brief states a deadline of **8 November 2026 at 23:59**.

For an explanation of the code and experiments to perform yourself, read [LEARNING_GUIDE.md](LEARNING_GUIDE.md).

# ML Assignment 2 — Breast Cancer Classification with Streamlit

**Name:** Akanksha Khot
**ID:** 2025AC05508
**Course:** M.Tech (AIML) — Machine Learning, BITS Pilani WILP

## a. Problem Statement

The goal of this assignment is to implement and compare multiple classification
models on a real-world dataset, then deploy an interactive Streamlit web
application that lets a user upload test data, select a model, and view its
evaluation metrics along with a confusion matrix / classification report. The
underlying task is **binary classification**: predicting whether a breast tumor
is **malignant** or **benign** based on measurements taken from a digitized image
of a fine needle aspirate (FNA) of a breast mass.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (accessed here via scikit-learn's
  built-in `load_breast_cancer()` loader, which packages the identical UCI dataset)
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric, real-valued features (≥ 12 required ✅) — computed from
  digitized images of cell nuclei, e.g. `mean radius`, `mean texture`,
  `mean perimeter`, `mean area`, `mean smoothness`, `mean concavity`,
  `worst radius`, `worst texture`, etc. (10 real-valued measurements, each
  reported as mean, standard error, and "worst"/largest value → 30 features)
- **Target variable:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign
- **Train/test split:** 80% train / 20% test, stratified by class, `random_state=42`
- **Preprocessing:** All features standardized with `StandardScaler` (fit on
  training data only, applied to test data)

## c. GitHub Repository Link

> https://github.com/akankshakhot/ML_ASSIGNMENT_2

## d. Models Used

Five classification models were trained on the same dataset and the same
train/test split, so results are directly comparable:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN, k=5)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble, 200 trees)

### Comparison Table (evaluation on the held-out test set, n=114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(These values are produced by `model/train_models.py`, which is deterministic
given `random_state=42`. Re-running it will reproduce this exact table.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset — the classes are close to linearly separable after standardization, so a simple linear decision boundary generalizes very well. Highest accuracy, precision, recall, F1, and MCC of all five models. |
| Decision Tree | Weakest performer. A single unpruned tree overfits the training data and does not generalize as well; lower accuracy and the lowest AUC/MCC indicate its probability estimates and error trade-offs are noisier than the other models. |
| kNN | Strong performance after feature scaling — distance-based similarity works well since the 30 features are all continuous and standardized. Ties with Random Forest on accuracy, F1 and MCC, and does so with a much simpler, non-parametric approach. |
| Naive Bayes | Reasonable accuracy but a comparatively lower F1/MCC than LR, kNN, and RF. This is expected: Naive Bayes assumes feature independence, which is violated here since many of the 30 features (mean, SE, and worst versions of the same measurement) are highly correlated. Interestingly its AUC is still high, meaning its ranking of predictions is good even though the default 0.5 threshold is sub-optimal. |
| Random Forest (Ensemble) | Second-best overall, and the ensemble smooths out the overfitting problem seen in the single Decision Tree — it matches kNN on accuracy/F1/MCC and has a near-top AUC, confirming that bagging many trees substantially improves on a single tree. |
| **Overall Winner for this dataset** | **Logistic Regression** — it achieves the top score on every single metric (Accuracy 0.9825, AUC 0.9954, Precision/Recall/F1 0.9861, MCC 0.9623), showing that once the features are standardized, a simple, well-regularized linear model is very hard to beat on this particular dataset. |

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt           # Python dependencies
│-- README.md                  # This file
│-- test_data.csv              # Held-out test set (features + true target) used in experiments
│-- model/
│   │-- train_models.py        # Trains all 5 models, computes metrics, saves artifacts (.py)
│   │-- ML_Assignment2.ipynb   # Same workflow as a Jupyter notebook, with EDA + plots (.ipynb)
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl             # StandardScaler fit on training data
│   │-- feature_names.pkl      # Expected feature column order
│   └-- model_comparison.csv   # Metrics table (same numbers as above)
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py     # optional: regenerate models/metrics
streamlit run app.py
```

## How This Was Deployed

1. Code pushed to a public GitHub repository (see link above).
2. Deployed via [Streamlit Community Cloud](https://streamlit.io/cloud):
   - Sign in with GitHub → **New app** → select this repo → branch `main` →
     main file `app.py` → **Deploy**.
3. Live app link: https://mlassignment2-llrypp7bw2dypxlunzchnj.streamlit.app/

## Streamlit App Features

- **Dataset upload (CSV):** Upload your own test CSV (30 feature columns + `target`),
  or the app falls back to the bundled `test_data.csv`.
- **Model selection dropdown:** Choose between Logistic Regression, Decision Tree,
  kNN, Naive Bayes, and Random Forest.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, MCC computed
  live on the uploaded data.
- **Confusion matrix & classification report:** Visual heatmap plus a full
  per-class precision/recall/F1 report.
- **Bonus:** Full 5-model comparison table (training-time evaluation) shown at
  the bottom of the app.

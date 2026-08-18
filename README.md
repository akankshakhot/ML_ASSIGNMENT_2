# ML Assignment 2 — Breast Cancer Classification with Streamlit

**Name:** Akanksha Khot
**ID:** 2025AC05508
**Course:** M.Tech (AIML), BITS Pilani WILP — Machine Learning

## a. Problem Statement

This assignment is about building and comparing a few different classification
models on the same dataset, then wrapping them in a Streamlit app so someone
can upload test data, pick a model, and see how it performs. The actual
prediction task is binary classification — given measurements from a
digitized image of a breast mass (a fine needle aspirate), predict whether
it's malignant or benign.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository — loaded here through
  scikit-learn's built-in `load_breast_cancer()` function, which is just a
  convenient wrapper around the same UCI data.
- **Size:** 569 rows, well above the 500-instance minimum.
- **Features:** 30 numeric columns (also above the 12-feature minimum)
  describing cell nuclei — things like radius, texture, perimeter, area,
  smoothness and concavity. Each of these 10 base measurements is reported
  three ways (mean, standard error, and "worst"/largest value), which is how
  you get to 30 columns.
- **Target:** binary — 0 for malignant, 1 for benign.
- **Class split:** 212 malignant cases vs. 357 benign.
- **Train/test split:** 80/20, stratified so the class balance holds in both
  sets, `random_state=42` for reproducibility.
- **Preprocessing:** features were standardized with `StandardScaler`, fit
  only on the training data and then applied to the test set.

## c. GitHub Repository Link

> https://github.com/akankshakhot/ML_ASSIGNMENT_2

## d. Models Used

All five models were trained on the exact same train/test split, so the
comparison below is apples-to-apples:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (k=5)
4. Gaussian Naive Bayes
5. Random Forest (200 trees)

### Comparison Table (test set, n=114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(These numbers come straight out of `model/train_models.py` — since
`random_state` is fixed, running it again reproduces the same table.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This came out on top across every metric. Once the features are scaled, the two classes are almost linearly separable, so a fairly simple model does the job better than the fancier ones. |
| Decision Tree | The weakest of the five. A single tree with no depth limit tends to memorize the training data rather than learn general patterns, which shows up as the lowest AUC and MCC here. |
| kNN | Did surprisingly well after scaling the features — makes sense, since it's entirely based on distance between points, and standardized data plays nicely with that. Ended up tied with Random Forest on accuracy, F1 and MCC. |
| Naive Bayes | Decent, but not the strongest. It assumes the features are independent of each other, which isn't really true here — a lot of these 30 columns are just different versions of the same underlying measurement (mean, standard error, worst). Interestingly the AUC is still quite high, so it's ranking predictions well even if the accuracy at the default threshold isn't as strong. |
| Random Forest (Ensemble) | A clear improvement over the single Decision Tree — averaging across 200 trees clearly helps with the overfitting problem. Landed in a virtual tie with kNN and close behind Logistic Regression. |
| **Overall winner for this dataset** | **Logistic Regression**, by a clear margin on every metric (Accuracy 0.9825, AUC 0.9954, F1 0.9861, MCC 0.9623). For this particular dataset, a simple linear model turned out to be hard to beat. |

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt           # Python dependencies
│-- README.md                  # This file
│-- test_data.csv              # Held-out test set used in experiments
│-- model/
│   │-- train_models.py        # Trains all 5 models, computes metrics
│   │-- ML_Assignment2.ipynb   # Same workflow as a Jupyter notebook, with EDA + plots
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl             # StandardScaler fit on training data
│   │-- feature_names.pkl      # Expected feature column order
│   └-- model_comparison.csv   # Metrics table
```

## Running It Locally

```bash
pip install -r requirements.txt
python model/train_models.py     # optional, regenerates the models and metrics
streamlit run app.py
```

## Deployment

The code lives in this public GitHub repo, and the app itself is deployed on
Streamlit Community Cloud — signed in with GitHub, pointed it at this repo's
main branch and `app.py`, and deployed.

**Live app:** https://mlassignment2-llrypp7bw2dypxlunzchnj.streamlit.app/

## What the App Does

- Lets you upload a test CSV (30 feature columns plus a `target` column), or
  just use the bundled `test_data.csv` if you don't have your own.
- Has a dropdown to switch between the five models.
- Shows Accuracy, AUC, Precision, Recall, F1 and MCC for whichever model is
  selected, computed live on the uploaded data.
- Displays a confusion matrix and a full classification report.
- Also includes a bonus comparison table showing all five models side by
  side, based on the original training-time evaluation.

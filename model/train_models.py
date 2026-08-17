"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates them on a held-out test set, saves trained models + scaler as pickle files,
and exports the test data (features + true label) as test_data.csv for use in the
Streamlit app.

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI ML Repository / scikit-learn built-in loader (sklearn.datasets.load_breast_cancer)
Instances: 569  |  Features: 30 (numeric, real-valued)  |  Task: Binary classification
Target: 0 = malignant, 1 = benign
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")
print(f"Features: {len(feature_names)} | Instances: {X.shape[0]}")

# ---------------------------------------------------------------------
# 2. Train / test split (stratified)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
os.makedirs(os.path.join(HERE), exist_ok=True)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save each trained model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    with open(os.path.join(HERE, fname), "wb") as f:
        pickle.dump(model, f)

# Save the scaler (needed by the Streamlit app to transform uploaded test data)
with open(os.path.join(HERE, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save feature name list (needed to validate uploaded CSVs)
with open(os.path.join(HERE, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

# ---------------------------------------------------------------------
# 5. Save results table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)
print("\nComparison table:\n", results_df)

# ---------------------------------------------------------------------
# 6. Save test data (features + true target) for the Streamlit app
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with shape {test_df.shape} to {ROOT}")

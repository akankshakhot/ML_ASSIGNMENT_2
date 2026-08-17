"""
app.py — Streamlit app for BITS Pilani ML Assignment 2
Breast Cancer Wisconsin (Diagnostic) — Binary Classification Demo

Features:
  a. Dataset upload option (CSV) — upload test data
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix / classification report
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 — Classification Demo",
    page_icon="🧬",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    """Load all trained models, the scaler, and the expected feature list."""
    models = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    comparison_df = pd.read_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))
    return models, scaler, feature_names, comparison_df


models, scaler, feature_names, comparison_df = load_artifacts()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🧬 Breast Cancer Classification — Model Comparison App")
st.markdown(
    """
This app demonstrates **5 classification models** trained on the
**Breast Cancer Wisconsin (Diagnostic) dataset** (30 features, 569 instances, binary target).
Upload a test CSV (features + a `target` column), pick a model, and see its performance.
"""
)

# ---------------------------------------------------------------------
# a. Dataset upload
# ---------------------------------------------------------------------
st.header("1. Upload Test Data (CSV)")
st.caption(
    "Upload a CSV containing the 30 feature columns plus a `target` column "
    "(0 = malignant, 1 = benign). You can use the provided `test_data.csv` from the repo."
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded yet — using the bundled sample `test_data.csv` for the demo.")
    sample_path = os.path.join(HERE, "test_data.csv")
    df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None

if df is not None:
    st.write("Preview of loaded data:")
    st.dataframe(df.head(), use_container_width=True)

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols[:5]}...")
        st.stop()
    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a `target` column with true labels (0/1).")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]

    # ---------------------------------------------------------------------
    # b. Model selection dropdown
    # ---------------------------------------------------------------------
    st.header("2. Select a Model")
    model_choice = st.selectbox("Choose a classification model", list(models.keys()))
    model = models[model_choice]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # ---------------------------------------------------------------------
    # c. Evaluation metrics
    # ---------------------------------------------------------------------
    st.header("3. Evaluation Metrics")

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")  # only one class present in uploaded data
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # ---------------------------------------------------------------------
    # d. Confusion matrix + classification report
    # ---------------------------------------------------------------------
    st.header("4. Confusion Matrix & Classification Report")

    col_a, col_b = st.columns(2)

    with col_a:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malignant (0)", "Benign (1)"],
                    yticklabels=["Malignant (0)", "Benign (1)"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)

    with col_b:
        report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"], zero_division=0)
        st.text("Classification Report")
        st.code(report)

    # ---------------------------------------------------------------------
    # Bonus: full model comparison table (all 5 models, precomputed on the
    # original held-out test split from training)
    # ---------------------------------------------------------------------
    st.header("5. All-Model Comparison (from training-time evaluation)")
    st.dataframe(comparison_df, use_container_width=True)

else:
    st.warning("Please upload a CSV file to proceed, or ensure test_data.csv is present in the repo.")

st.markdown("---")
st.caption("BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning — Assignment 2")
st.caption("Akanksha Khot | 2025AC05508")

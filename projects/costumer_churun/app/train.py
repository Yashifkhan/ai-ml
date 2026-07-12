import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ------------------ DATA LOADING & CLEANING ------------------

df = pd.read_csv("../data/Customer-Churn.csv")
df = df.drop("customerID", axis=1)

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

updition_col = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Churn'
]
for col in updition_col:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod'], drop_first=True)

df["MultipleLines"] = df["MultipleLines"].fillna(0)
df["InternetService"] = df["InternetService"].fillna(0)

internet_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
df[internet_cols] = df[internet_cols].fillna(0)

df = pd.get_dummies(df, columns=["InternetService"], drop_first=True)

df.drop_duplicates(inplace=True)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# ------------------ NEW: FEATURE ENGINEERING ------------------

# Tenure buckets — tenure is usually the strongest churn predictor,
# bucketing helps linear/tree models capture non-linear drop-off patterns
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 60, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
)
df = pd.get_dummies(df, columns=["tenure_group"], drop_first=True)

# Average monthly spend so far — captures spending intensity relative to tenure
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# ------------------ TRAIN/TEST SPLIT ------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ HELPER: EVALUATE WITH CUSTOM THRESHOLD ------------------

def evaluate_model(name, y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    print(f"\n--- {name} (threshold={threshold}) ---")
    print("Accuracy:", accuracy_score(y_true, preds))
    print("Precision:", precision_score(y_true, preds))
    print("Recall:", recall_score(y_true, preds))
    print("F1 Score:", f1_score(y_true, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_true, preds))
    return preds

def find_best_threshold(y_true, probs, metric="f1"):
    best_thresh, best_score = 0.5, 0
    for t in np.arange(0.2, 0.6, 0.02):
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds) if metric == "f1" else recall_score(y_true, preds)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score

# ------------------ LOGISTIC REGRESSION ------------------

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # key fix: was missing before
    random_state=42
)
log_model.fit(X_train_scaled, y_train)

log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

print("=== Logistic Regression @ default 0.5 threshold ===")
evaluate_model("Logistic Regression", y_test, log_probs, threshold=0.5)

best_t_log, best_f1_log = find_best_threshold(y_test, log_probs, metric="f1")
print(f"\nBest threshold for LR (max F1): {best_t_log:.2f}")
evaluate_model("Logistic Regression", y_test, log_probs, threshold=best_t_log)

# ------------------ RANDOM FOREST (TUNED) ------------------

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample"]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring="f1",     # optimize for f1 instead of raw accuracy
    cv=5,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("\nBest RF params:", grid.best_params_)
rf_model = grid.best_estimator_

rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest @ default 0.5 threshold ===")
evaluate_model("Random Forest", y_test, rf_probs, threshold=0.5)

best_t_rf, best_f1_rf = find_best_threshold(y_test, rf_probs, metric="f1")
os.makedirs("../models", exist_ok=True)

# Save Logistic Regression model
joblib.dump(log_model, "../models/logistic_regression_model.pkl")

# Save Random Forest model
joblib.dump(rf_model, "../models/random_forest_model.pkl")

# Save scaler (needed for LR at inference time)
joblib.dump(scaler, "../models/scaler.pkl")

# Save feature column order (CRITICAL — so predict.py builds input in same shape/order)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "../models/feature_columns.pkl")

# Save best thresholds found for each model (so predict.py doesn't default to 0.5)
thresholds = {
    "logistic_regression": best_t_log,
    "random_forest": best_t_rf
}
joblib.dump(thresholds, "../models/thresholds.pkl")

print("\nAll models and artifacts saved to ../models/")
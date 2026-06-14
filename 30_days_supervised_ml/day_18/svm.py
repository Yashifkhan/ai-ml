import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Scale BEFORE training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------- Try Different Models -------- #

# Linear
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)

# Polynomial
model_poly = SVC(kernel='poly', degree=3)
model_poly.fit(X_train, y_train)
pred_poly = model_poly.predict(X_test)

# RBF
model_rbf = SVC(kernel='rbf', C=10, gamma=0.01)
model_rbf.fit(X_train, y_train)
pred_rbf = model_rbf.predict(X_test)

# -------- Results -------- #
print("Linear Accuracy:", accuracy_score(y_test, pred_linear))
print("Polynomial Accuracy:", accuracy_score(y_test, pred_poly))
print("RBF Accuracy:", accuracy_score(y_test, pred_rbf))

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

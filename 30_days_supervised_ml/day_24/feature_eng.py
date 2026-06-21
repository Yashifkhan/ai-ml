import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2, f_classif

df=pd.read_csv("Telco_Customer_Churn.csv")

# Remove unnecessary columns
df.drop(columns=["customerID"], inplace=True, errors='ignore')

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)


X = df.drop("Churn_Yes", axis=1)   # change target name
y = df["Churn_Yes"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

corr = df.corr()

# print(corr)
# correlation with target
# target_corr = corr["target"].sort_values(ascending=False)
# print(target_corr)

# print(df.head())
# print(df.info())
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("churn_data.csv")
print(df.head())

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['ContractType'] = df['ContractType'].map({'Monthly': 0, 'Yearly': 1})
df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber': 1})
df['TechSupport'] = df['TechSupport'].map({'No': 0, 'Yes': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']



# simple direct train test split 
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # apply cross validation kfold 
# model = LogisticRegression(max_iter=200)
# scores = cross_val_score(model, X, y, cv=5)


# apply stratifield k-fold 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # maintain class balance
    random_state=42
)

model = LogisticRegression(max_iter=200)

skf = StratifiedKFold(n_splits=5)

scores = cross_val_score(model, X_train, y_train, cv=skf)

# print("Stratified CV Scores:", scores)
# print("Mean CV Accuracy:", scores.mean())


# print("CV Scores:", scores)
# print("Mean CV Accuracy:", scores.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Final Test Accuracy:", accuracy_score(y_test, y_pred))
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Train-Test Accuracy:", accuracy_score(y_test, y_pred))
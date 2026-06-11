import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

requred_featured=["Survived","Age","Sex","Pclass","Fare","Embarked"]
df=pd.read_csv("Titanic-Dataset.csv",usecols=requred_featured)

df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.isnull().sum())

# convert the cat into binary 
lr=LabelEncoder()
df["Sex"]=lr.fit_transform(df["Sex"])
df["Embarked"]=lr.fit_transform(df["Embarked"])

X=df.drop(["Survived"],axis=1)
y=df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr=LogisticRegression(max_iter=1000)
lr.fit(X,y)
y_predict_lr=lr.predict(X_test)


# KNN algorithm 
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_predict_knn=knn.predict(X_test)

# Decision Tree 
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_predict_dt=dt.predict(X_test)


def evaluate(y_test, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    
evaluate(y_test, y_predict_lr, "Logistic Regression")
evaluate(y_test, y_predict_knn, "KNN")
evaluate(y_test, y_predict_dt, "Decision Tree")

# print(df.head())
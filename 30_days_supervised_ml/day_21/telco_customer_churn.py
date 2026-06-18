import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report ,precision_score,recall_score,f1_score

required_features=[
"gender",
"SeniorCitizen",
"Partner",
"Dependents",
"tenure",
"PhoneService",
"MultipleLines",
"InternetService",
"OnlineSecurity",
"OnlineBackup",
"DeviceProtection",
"TechSupport",
"StreamingTV",
"StreamingMovies",
"Contract",
"PaperlessBilling",
"PaymentMethod",
"MonthlyCharges",
"TotalCharges",
"Churn",
]
df = pd.read_csv("Telco-Customer-Churn.csv", usecols=required_features)

# this way is use for thebegginer insted of we use the one hotencoding 
# df["gender"]=df["gender"].map({"Female":0,"Male":1})
# df["Partner"]=df["Partner"].map({"No":0,"Yes":1})
# df["Dependents"]=df["Dependents"].map({"No":0,"Yes":1})
# df["PhoneService"]=df["PhoneService"].map({"No":0,"Yes":1})

# one hot encoding 
# df["Partner"] = pd.get_dummies(df["Partner"], drop_first=True).astype(int)

# for all binary column categorical --> numeric but value are same 
binary_columns=["Partner","Dependents","PhoneService","PaperlessBilling","StreamingTV","StreamingMovies","OnlineSecurity","TechSupport","OnlineBackup","DeviceProtection","Churn"]
df[binary_columns]=df[binary_columns].replace({"Yes":1,"No":0})


# for gender column 
df["gender"] = pd.get_dummies(df["gender"], drop_first=True).astype(int)

# for the multi column categorical data
multi_columns=["Contract","PaymentMethod","InternetService"]
df=pd.get_dummies(df,columns=multi_columns,drop_first=True)

df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
df["MultipleLines"] = df["MultipleLines"].map({"No":0, "Yes":1})

# alter nativve of this select data type 
# print the spacific column value 
rest_column=["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","TotalCharges"]


# rest_column=df.select_dtypes(include="object").columns

for col in rest_column:
    df[rest_column] = df[rest_column].replace("No internet service", "No")
    df[rest_column] = df[rest_column].replace({"No":0, "Yes":1})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# check and fill the miss value 
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# df[['MonthlyCharges','TotalCharges','tenure']] = scaler.fit_transform(
#     df[['MonthlyCharges','TotalCharges','tenure']]
# )

# Feature Engineering   
df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

X = df.drop('Churn', axis=1)
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# print(df.head())
# print(df.info())

# train the model step by step 
# Feature Scaling scal the data high range to low 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Sample y_pred:", y_pred[:5])
    print("Type:", y_pred.dtype)

    import numpy as np
    if not np.array_equal(y_pred, y_pred.astype(int)):
        print("⚠️ Converting continuous to binary")
        y_pred = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, pre, rec, f1
# store all model 
results = []
# 1 model Logistic Regression 
lr=LogisticRegression()
lr.fit(X_train_scaled,y_train)
acc, pre, rec, f1 = evaluate_model(lr, X_test_scaled, y_test)
results.append(["Logistic Regression", acc, pre, rec, f1])
# 2. KNN 
knn=KNeighborsClassifier()
knn.fit(X_train_scaled,y_train)
acc, pre, rec, f1 = evaluate_model(knn, X_test_scaled, y_test)
results.append(["KNN", acc, pre, rec, f1])
# 3. Decision Tree
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
acc, pre, rec, f1 = evaluate_model(dt, X_test, y_test)
results.append(["Decision Tree", acc, pre, rec, f1])
# 4. Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
acc, pre, rec, f1 = evaluate_model(rf, X_test, y_test)
results.append(["Random Forest", acc, pre, rec, f1])
# 5. SVM
svm = SVC()
svm.fit(X_train_scaled, y_train)
acc, pre, rec, f1 = evaluate_model(svm, X_test_scaled, y_test)
results.append(["SVM", acc, pre, rec, f1])
# 6. Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
acc, pre, rec, f1 = evaluate_model(gb, X_test, y_test)
results.append(["Gradient Boosting", acc, pre, rec, f1])

# print(results)
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print(results_df)
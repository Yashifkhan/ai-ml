import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

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

# print(df.head())
# print(df.tail(10))
# print(df.columns.tolist())


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

# df.info()
# print("value count of chrun",df['Churn'].value_counts())

# print(df.head())
# print(df[rest_column])


# Feature Scaling

scaler = StandardScaler()

df[['MonthlyCharges','TotalCharges','tenure']] = scaler.fit_transform(
    df[['MonthlyCharges','TotalCharges','tenure']]
)


# Feature Engineering   
df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

X = df.drop('Churn', axis=1)
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

model = LogisticRegression()
model.fit(X_train, y_train)

sample_no_churn = [[
0,  # gender (Female)
0,  # SeniorCitizen
1,  # Partner
1,  # Dependents
48, # tenure (high → stable customer)
1,  # PhoneService
1,  # MultipleLines
1,  # OnlineSecurity
1,  # OnlineBackup
1,  # DeviceProtection
1,  # TechSupport
1,  # StreamingTV
1,  # StreamingMovies
0,  # PaperlessBilling (less risky)
55.0,   # MonthlyCharges (moderate)
3000.0, # TotalCharges (high lifetime value)

1,0,   # Contract (One year)
1,0,0, # PaymentMethod (Credit card automatic)
0,0 ,   # InternetService (DSL assumed)
0
]]

y_pred = model.predict(X_test)
y_predict=model.predict(sample_no_churn)
print("model predict value : ",y_predict[0])
print("Accuracy:", accuracy_score(y_test, y_pred))

print("model predict value : ",y_predict[0])
print("Accuracy:", accuracy_score(y_test, y_pred))


# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
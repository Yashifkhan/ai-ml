import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt

# use only requered featured 
requred_featurs=["tenure","MonthlyCharges","TotalCharges","Contract","InternetService","Churn"]
df=pd.read_csv("Telco-Customer-Churn.csv",usecols=requred_featurs)

# print(df.head())
# print(df.info())
# print(df.isnull().sum())

# target column is chrun
# so we convert into numaric value 
df["Churn"]=df["Churn"].map({"Yes":1,"No":0})

# convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = pd.get_dummies(df, columns=["InternetService", "Contract"], drop_first=True)


df = pd.get_dummies(df, drop_first=True)
df.dropna(inplace=True)

# split the data 
X=df.drop(["Churn"],axis=1)
y=df["Churn"]


# train test split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2, random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


model=LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# print("y_pred",y_pred[0])
# print("y_pred",y_pred[2])

for i in range(2):
    print("Prediction:", y_pred[i])
    print("Probability:", model.predict_proba([X_test[i]]))
    # print("------")
    prob = model.predict_proba([X_test[i]])[0][1]
    if prob > 0.7:
        print("High Risk Customer 🚨")
    elif prob > 0.4:
        print("Medium Risk ⚠️")
    else:
        print("Safe Customer ✅")
        
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# print(df.head())



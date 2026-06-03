# Loan Approval Prediction
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
df = pd.DataFrame({
    "income": [20,30,40,50,60,70,80,90],
    "credit": [300,400,500,600,650,700,750,800],
    "age":    [22,25,30,35,40,45,50,55],
    "approved":[0,0,0,1,1,1,1,1]
})

X = df[["income","credit","age"]]
y = df["approved"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)



# sample = [[45, 300, 32]]
samples = [
    [25, 350, 23],
    [45, 550, 32],
    [85, 780, 50]
]

# Predict
# y_pred = model.predict(sample)
# y_prob = model.predict_proba(sample)

# print(y_pred)

threshold = 0.5 
# prob = y_prob[0][1] 

# if prob >= threshold:
#     print("Approved the prob. of :" ,prob)
# else:
#     print("Not Approved the prob. of : ",prob)


for s in samples:
    prob = model.predict_proba([s])[0][1]

    if prob >= 0.5:
        print(f"{s} → Approved ({prob:.2f})")
    else:
        print(f"{s} → Not Approved ({prob:.2f})")

# Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
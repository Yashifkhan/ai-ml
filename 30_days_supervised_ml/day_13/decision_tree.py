# # Loan Approval Model project 
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# data = {
#     "income": [20,30,40,50,60,70,80,90],
#     "credit": [300,400,500,600,650,700,750,800],
#     "age":    [22,25,30,35,40,45,50,55],
#     "loan_approved": [0,0,0,1,1,1,1,1]
# }

# df=pd.DataFrame(data)


# X = df[["income", "credit", "age"]]
# y = df["loan_approved"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = DecisionTreeClassifier(max_depth=3)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# sample = [[45, 450, 42]]  # income, credit, age
# result = model.predict(sample)

# if result[0] == 1:
#     print("Loan Approved")
# else:
#     print("Loan Rejected")
# # print(df.head())

# build the proper project 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("Loan_default.csv")

df['Education'] = df['Education'].map({
    'High School':0, 'Bachelor':1, 'Master':2
})
df = df.drop("LoanID", axis=1)
df.isnull().sum()
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Default", axis=1)
y = df["Default"]

model = DecisionTreeClassifier(max_depth=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

sample = pd.DataFrame([{
    "Age": 30,
    "Income": 50000,
    "LoanAmount": 20000,
    "CreditScore": 700,
    "MonthsEmployed": 48,
    "NumCreditLines": 4,
    "InterestRate": 10,
    "LoanTerm": 36,
    "DTIRatio": 0.3,
    "Education": "Bachelor",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Home",
    "HasCoSigner": "Yes"
}])
sample2 = [[
    22,
    20000,
    30000,
    500,
    6,
    1,
    18,
    60,
    0.6,
    0,
    0,
    0,
    0,
    1,
    0,
    0
]]

print(model.predict(sample))
# print(model.predict(sample2))
# print(df.columns.tolist())
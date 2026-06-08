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

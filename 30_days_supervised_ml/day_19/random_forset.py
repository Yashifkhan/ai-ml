# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier

# df = pd.DataFrame({
#     "Age": [25,45,35,50,23],
#     "Salary": [30000,80000,50000,90000,20000],
#     "CreditScore": [600,750,700,800,580],
#     "LoanApproved": [0,1,1,1,0]
# })

# sample1=df.sample(n=5,replace=True,random_state=1)
# sample2=df.sample(n=5,replace=True,random_state=2)
# sample3=df.sample(n=5,replace=True,random_state=3)


# X = df.drop("LoanApproved", axis=1)
# y = df["LoanApproved"]

# trees = []

# for i in range(3):
#     # Step 1: Bootstrap sample
#     sample = df.sample(n=len(df), replace=True)

#     X_sample = sample.drop("LoanApproved", axis=1)
#     y_sample = sample["LoanApproved"]

#     # Step 2: Train tree
#     tree = DecisionTreeClassifier(max_depth=3)
#     tree.fit(X_sample, y_sample)

#     # Step 3: Store tree
#     trees.append(tree)

# # Step 4: Now check predictions
# for i, tree in enumerate(trees):
#     print(f"Tree {i} prediction:", tree.predict(X))

# # print(trees)

# # Collect predictions from all trees
# all_preds = []

# for tree in trees:
#     preds = tree.predict(X)
#     all_preds.append(preds)
    
#     # Convert to array
# all_preds = np.array(all_preds)

# # Majority voting
# final_preds = []

# for i in range(len(X)):
#     votes = all_preds[:, i]
#     final = np.bincount(votes).argmax()
#     final_preds.append(final)

# print("Final Prediction:", final_preds)

# # print("Tree 1 Data:\n", sample1)
# # print("Tree 2 Data:\n", sample2)
# # print("Tree 3 Data:\n", sample3)


# # follow this step in random forest  
# #  Bootstrap Sampling (different datasets)

# # 2. Train multiple decision trees

# # 3. Random feature selection at each split

# # 4. Each tree builds its own logic

# # 5. Each tree gives prediction

# # 6. Combine predictions:
# #    - Classification → Voting
# #    - Regression → Average



# learn proper random forest algo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("loan_data.csv")



X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

print(feature_importance.sort_values(by="Importance", ascending=False))

new_data = [[30, 60000, 710, 260000]]

prediction = model.predict(new_data)

print("Loan Approved:", prediction)

# print("Predictions:", y_pred)
# print("Accuracy:", accuracy)

# print(df.head())


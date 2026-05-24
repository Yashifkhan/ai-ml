# print("prectices of data preprocessing")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# data = pd.DataFrame({
#     "Age": [25, None, 30, 22, 40, 25],
#     "Salary": [50000, 60000, None, 45000, 80000, 50000],
#     "Gender": ["Male", "Female", "Male", "Female", "Male", "Male"],
#     "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi"],
#     "Purchased": ["No", "Yes", "No", "Yes", "No", "No"]
# })

# print(data)

# # handling the missing value in data 
# data["Age"]=data["Age"].fillna(data["Age"].median())
# data["Salary"]=data["Salary"].fillna(data["Salary"].mean())


# remove the duplicates row in data
# data=data.drop_duplicates()


# Encoded categorical data ,convert categorical value in number 
# data["Gender"]=data["Gender"].map({"Male": 1,"Female":0})
# data["Purchased"]=data["Purchased"].map({"Yes":1,"No":0})
# data=pd.get_dummies(data,columns=["City"],drop_first=True,dtype=int)

# split the feature and target 
# X=data.drop("Purchased",axis=1)
# y=data["Purchased"]
# # print(X)

# data clean complete 
# train test split ,for train and test 
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)


# 6. Feature scaling 
# convert the value in also 1,0 mean is 0 and std is 1
# Age: [20, 30, 40]-->>[-1, 0, +1] 
# scaler=StandardScaler()
# X_train_scaled=scaler.fit_transform(X_train)
# X_test_scaled=scaler.transform(X_test)


# print("X train data")
# print(X_train_scaled)
# print("X test data")
# print(X_test_scaled)



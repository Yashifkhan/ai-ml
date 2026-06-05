# print("i want to learn the knn")
# 
# without sk learn algorithm 
# data = [
#     (100, "apple"),
#     (110, "apple"),
#     (120, "apple"),
#     (150, "orange"),
#     (160, "orange"),
#     (170, "orange")
# ]

# distance=[]
# new_weight=130
# k=3
# for weight ,label in data:
#     dist=abs(weight - new_weight)
#     distance.append((dist,label))

# distance.sort()
# k_nearest=distance[:3]

# votes={}
# for _, label in k_nearest:
#     votes[label]=votes.get(label,0)+1
    
# prediction = max(votes, key=votes.get)
# print("Prediction:", prediction)


# with algorith build features use 
# Fruit Classifier 
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import numpy as np


# # Features: weight
# X = np.array([[100], [110], [120], [150], [160], [170]])

# # Labels
# y = np.array(["apple", "apple", "apple", "orange", "orange", "orange"])


# model=KNeighborsClassifier(n_neighbors=3)
# model.fit(X,y)

# new_weight = int(input("Enter fruit weight b/w 100 to 200: "))
# prediction = model.predict([[new_weight]])
# print("Predicted fruit:", prediction[0])

# project 3 
# Customer Purchase Prediction
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# X = np.array([
#     [22, 20000],
#     [25, 25000],
#     [30, 30000],
#     [35, 50000],
#     [40, 60000],
#     [45, 65000]
# ])
# y = np.array([0, 0, 0, 1, 1, 1])  # 0 = No, 1 = Yes
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X, y)
# # New customer
# new_customer = [[33, 40000]]

# prediction = model.predict(new_customer)

# if prediction[0] == 1:
#     print("Customer will BUY")
# else:
#     print("Customer will NOT BUY")

# project 4 

# house priece prediction compare with anothe hosue 
import pandas as pd
features = [
    "Gr Liv Area",     # instead of GrLivArea
    "Bedroom AbvGr",   # instead of BedroomAbvGr
    "Full Bath",       # instead of FullBath
    "Garage Cars",     # instead of GarageCars
    "Year Built"       # instead of YearBuilt
]
df=pd.read_csv("AmesHousing.csv",usecols=features)

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
X = df[features]
y = df["SalePrice"]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

print(X.isnull().sum())

X = X.fillna(X.mean())

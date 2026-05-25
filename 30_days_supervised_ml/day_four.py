# import pandas as pd 
# from sklearn.linear_model import LinearRegression

# data = pd.DataFrame({
#     "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     "marks": [35, 40, 50, 55, 60, 65, 70, 75, 85, 95]
# })


# X=data[["hours"]]
# y=data["marks"]
# model=LinearRegression()
# model.fit(X,y)

# predect_value=model.predict([[12]])
# print("Model predict value is : ",predect_value)


# prectice 2 
import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

data=fetch_california_housing()

df=pd.DataFrame(data.data)
df.columns=data.feature_names


X=df
y=data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()

X_train=scaler.fit(X_train)
X_test=scaler.fit_transform(X_test)
# Train the model
regression=LinearRegression()
regression.fit(X_train,y_train)
mse=cross_val_score(regression,X_train,y_train,scoring="neg_mean_squared_error",cv=5)

reg_pred=regression.predict(X_test)
# print(df.head())
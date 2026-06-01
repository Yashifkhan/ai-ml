import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df=pd.read_csv("student_exam_scores.csv")


# check the data 
# print(df.isnull().sum())


# plot the data for visilation 
# sns.scatterplot(x=df["hours_studied"],y=df["exam_score"])
# plt.show()

df=df.drop(["student_id"],axis=1)
print(df.head())
# spliut the data 
X=df.drop("exam_score",axis=1)
y=df["exam_score"]

# Train Test Split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.2,random_state=42
)

# Train Model
model=LinearRegression()
model.fit(X_train,y_train)


# test for one student 
# test_row = [[8.0, 6.5, 85.0, 70,]]
# Custom prediction
test_data = [[1.0, 10, 95.5, 80.0]]



y_custom = model.predict(test_data)
print("Predicted score:", y_custom[0])


# Predict
y_predict=model.predict(test_data)

# R2
r2 = r2_score(y_test, y_predict)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

print("Model accurecy", r2)
# print("RMSE:", rmse)

print("predicted score of student : ",y_predict[0])



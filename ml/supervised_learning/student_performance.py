import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import r2_score

X=np.array([1,2,3,4,5]).reshape(-1, 1)
y=np.array([20,40,60,80,100])

model=LinearRegression()
model.fit(X,y)

# Correct accuracy calculation
y_pred = model.predict(X)
accurecy = r2_score(y, y_pred)

# New prediction
predict_value = model.predict([[6]])

print("accuracy of model:", accurecy)
print("result of model:", predict_value)
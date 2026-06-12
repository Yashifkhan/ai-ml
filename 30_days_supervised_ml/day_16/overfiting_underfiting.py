# underfiting model 
# 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# # simple curve data
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)

X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model = LinearRegression()
# model.fit(X_train, y_train)

# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

# print("Train Error:", mean_squared_error(y_train, train_pred))
# print("Test Error:", mean_squared_error(y_test, test_pred))



# overfiting model 
# model = make_pipeline(
#     PolynomialFeatures(degree=15),
#     LinearRegression()
# )

# model.fit(X_train, y_train)

# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

# print("Train Error:", mean_squared_error(y_train, train_pred))
# print("Test Error:", mean_squared_error(y_test, test_pred))


# good fit model 
model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train Error:", mean_squared_error(y_train, train_pred))
print("Test Error:", mean_squared_error(y_test, test_pred))
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


df=pd.read_csv("house_price_regression_learning.csv")


# fix invalid
df.loc[df["house_age_years"] <= 0, "house_age_years"] = None
df.loc[df["area_sqft"] <= 0, "area_sqft"] = np.nan

# fill missing
df["area_sqft"] = df["area_sqft"].fillna(df["area_sqft"].median())
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
df["location_score"] = df["location_score"].fillna(df["location_score"].median())
df["house_age_years"] = df["house_age_years"].fillna(df["house_age_years"].median())

# Apply Transformation
df["area_sqft"], area_lambda = stats.boxcox(df["area_sqft"])
df["living_space_index"], lsi_lambda = stats.boxcox(df["living_space_index"])

# Feature Scaling
X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

new_house = pd.DataFrame([{
    "area_sqft": 2000,
    "bedrooms": 3,
    "living_space_index": 180,
    "location_score": 7.5,
    "house_age_years": 10
}])

new_house["area_sqft"] = stats.boxcox(new_house["area_sqft"], lmbda=area_lambda)
new_house["living_space_index"] = stats.boxcox(new_house["living_space_index"], lmbda=lsi_lambda)

new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print("Predicted Price:", predicted_price[0])


# TRAIN METRICS
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# TEST METRICS (REAL PERFORMANCE)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("---- TRAIN ----")
print(train_mae, train_mse, train_rmse, train_r2)

print("---- TEST ----")
print(test_mae, test_mse, test_rmse, test_r2)

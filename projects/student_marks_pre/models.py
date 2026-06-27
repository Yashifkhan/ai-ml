import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

df=pd.read_csv("students_final_raw (1).csv")

# remove the column which is not need 
uselesscolumns=["Math_marks", "Science_marks", "Social_Science_marks", "English_marks", "Hindi_marks","Total_marks", "core_ability", "data_source","goout", "Dalc", "Walc",'Average_marks']

X=df.drop(columns=uselesscolumns,axis=1)
y=df["Average_marks"]


# handle the Categorical Encoding 
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
               'schoolsup', 'famsup', 'paid', 'activities', 
               'nursery', 'higher', 'internet', 'romantic']

multi_cols = ['Mjob', 'Fjob', 'reason', 'guardian']

numeric_cols = [col for col in X.columns if col not in binary_cols + multi_cols]


# Step 2: ColumnTransformer banao
preprocessor = ColumnTransformer(
    transformers=[
        ('binary_enc', OneHotEncoder(drop='if_binary'), binary_cols),
        ('multi_enc', OneHotEncoder(), multi_cols)
    ],
    remainder='passthrough'   # numeric columns ko as-is rakhega
)

X_transformed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, 
    test_size=0.2, 
    random_state=42
)

# train the model 

# first linear regression 
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# sample_student = pd.DataFrame([{
#     'school': 'GP',
#     'sex': 'F',
#     'age': 16,
#     'address': 'U',
#     'famsize': 'GT3',
#     'Pstatus': 'T',
#     'Medu': 3,
#     'Fedu': 2,
#     'Mjob': 'teacher',
#     'Fjob': 'other',
#     'reason': 'course',
#     'guardian': 'mother',
#     'traveltime': 1,
#     'studytime': 3,
#     'failures': 0,
#     'schoolsup': 'no',
#     'famsup': 'yes',
#     'paid': 'no',
#     'activities': 'yes',
#     'nursery': 'yes',
#     'higher': 'yes',
#     'internet': 'yes',
#     'romantic': 'no',
#     'famrel': 4,
#     'freetime': 3,
#     'health': 4,
#     'absences': 2,
#     'sleep_hours': 7.0,
#     'mobile_social_hours': 3.0,
#     'Math_self_rating': 5,
#     'Science_self_rating': 5,
#     'Social_Science_self_rating': 5,
#     'English_self_rating': 5,
#     'Hindi_self_rating': 5
# }])

# # Test set performance
# mae_lr = mean_absolute_error(y_test, y_pred)
# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
# r2_lr_test = r2_score(y_test, y_pred)
# # Training set performance (overfitting check)
# y_pred_train_lr = model.predict(X_train)
# r2_lr_train = r2_score(y_train, y_pred_train_lr)
# print("=== Linear Regression Performance ===")
# print("MAE (test):", mae_lr)
# print("RMSE (test):", rmse_lr)
# print("R2 (test):", r2_lr_test)
# print("R2 (train):", r2_lr_train)


# Decision Tree Regressor 
# dt_model = DecisionTreeRegressor(max_depth=4,random_state=42)
# dt_model.fit(X_train, y_train)
# y_pred_dt = dt_model.predict(X_test)

# # Test set performance
# mae_dt = mean_absolute_error(y_test, y_pred_dt)
# rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
# r2_dt_test = r2_score(y_test, y_pred_dt)
# # Training set performance (overfitting check)
# y_pred_train_dt = dt_model.predict(X_train)
# r2_dt_train = r2_score(y_train, y_pred_train_dt)
# print("=== Decision Tree Performance ===")
# print("MAE (test):", mae_dt)
# print("RMSE (test):", rmse_dt)
# print("R2 (test):", r2_dt_test)
# print("R2 (train):", r2_dt_train)


# Random Forest Regressor 
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# y_pred_rf = rf_model.predict(X_test)
# y_pred_train_rf = rf_model.predict(X_train)

# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
# r2_rf_test = r2_score(y_test, y_pred_rf)
# r2_rf_train = r2_score(y_train, y_pred_train_rf)

# print("=== Random Forest Performance ===")
# print("MAE (test):", mae_rf)
# print("RMSE (test):", rmse_rf)
# print("R2 (test):", r2_rf_test)
# print("R2 (train):", r2_rf_train)



# train this model usig the pipeline 

# Pipeline 1: Linear Regression
pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Pipeline 2: Decision Tree (max_depth=4, jo best tha)
pipe_dt = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=4, random_state=42))
])

# Pipeline 3: Random Forest
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Teeno pipelines train karo
pipe_lr.fit(X_train_raw, y_train)
pipe_dt.fit(X_train_raw, y_train)
pipe_rf.fit(X_train_raw, y_train)


# for name, pipe in [('Linear Regression', pipe_lr), ('Decision Tree', pipe_dt), ('Random Forest', pipe_rf)]:
#     y_pred_pipe = pipe.predict(X_test_raw)
#     r2 = r2_score(y_test, y_pred_pipe)
#     mae = mean_absolute_error(y_test, y_pred_pipe)
#     print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}")

print("All 3 pipelines trained successfully!")

import joblib

joblib.dump(pipe_lr, 'linear_regression_pipeline.pkl')
joblib.dump(pipe_dt, 'decision_tree_pipeline.pkl')
joblib.dump(pipe_rf, 'random_forest_pipeline.pkl')

print("All 3 models dumped successfully!")
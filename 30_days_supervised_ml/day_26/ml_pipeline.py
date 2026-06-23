import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,classification_report

df=pd.read_csv("student_performance_dataset.csv")

df=df.drop("Student_ID",axis=1)
X=df.drop("Pass_Fail",axis=1)
y=df["Pass_Fail"]

print(df.head())
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# coloumn sparation for numaric and  categorical 
num_col=['Study_Hours_per_Week', 'Attendance_Rate', 'Final_Exam_Score']
cat_column=['Extracurricular_Activities']

# preprocessing 
preprocessor =ColumnTransformer([
    ("num",StandardScaler(),num_col),
    ("cat",OneHotEncoder(),cat_column)
])

# create pipeline 
pipeline=Pipeline([
    ("preprocessing",preprocessor),
    ('model',RandomForestClassifier(
    n_estimators=50,   # number of trees
    max_depth=3,        # control overfitting
    random_state=42
))
])

# train model 
# pipeline.fit(X_train,y_train)

# predict 

# print(y_predict)


test_data = pd.DataFrame([
    {
        "Gender": 1,
        "Study_Hours_per_Week": 0,
        "Attendance_Rate": 20.5,
        "Extracurricular_Activities": "Yes",
        "Education": "Bachelors",
        "Final_Exam_Score": 62
    },
    {
        "Gender": 0,
        "Study_Hours_per_Week": 15,
        "Attendance_Rate": 60.2,
        "Extracurricular_Activities": "No",
        "Education": "High School",
        "Final_Exam_Score": 45
    }
])


sample2 = pd.DataFrame([{
    "Gender": 1,
    "Study_Hours_per_Week": 40,
    "Attendance_Rate": 90.5,
    "Extracurricular_Activities": "Yes",
    "Education": "Bachelors",
    "Final_Exam_Score": 80
}])

# print(pipeline.predict(sample))
# print(pipeline.predict_proba(sample))
# test_pred = pipeline.predict(test_data.iloc[[0]])
# print("Manual Test Prediction:", test_pred)
# accurecy 
# print("accuracy",accuracy_score(y_test,y_predict))

# print(df.head())


# train the regression mode 

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(sample2)
print(y_pred)
print(pipeline.predict_proba(sample2)) 


# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
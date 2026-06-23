import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,classification_report
import joblib


df=pd.read_csv("python_learning_exam_performance.csv")
X = df.drop(["passed_exam", "final_exam_score"], axis=1)
y = df["passed_exam"]

print(df.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


cat_feature = [
    'country',
    'prior_programming_experience',
    'uses_kaggle',
    'participates_in_discussion_forums'
]
num_feature = [
    'age',
    'weeks_in_course',
    'hours_spent_learning_per_week',
    'practice_problems_solved',
    'projects_completed',
    'tutorial_videos_watched',
    'debugging_sessions_per_week',
    'self_reported_confidence_python',
]

preprocessor =ColumnTransformer([
    ("num",StandardScaler(),num_feature),
    ("cat",OneHotEncoder(),cat_feature)
])

pipeline_lr=Pipeline([
    ("processing",preprocessor),
    ("model",LogisticRegression())
])

pipeline_rf = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier())
])

pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# Logistic
pred_lr = pipeline_lr.predict(X_test)
print("Logistic Accuracy:", accuracy_score(y_test, pred_lr))

# Random Forest
pred_rf = pipeline_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

joblib.dump(pipeline_lr, "student_lr.pkl")
joblib.dump(pipeline_rf, "student_rf.pkl")

print("Models Saved!")

# print("predict value : " , y_predict)
# print(df.columns.tolist())
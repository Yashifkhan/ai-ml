import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, "data", "student-mat.csv")

cols = [
    "studytime",   # study_hours
    "absences",    # attendance
    "G1", "G2",    # previous_marks
    "internet",
    "Medu", "Fedu",
    "G3",
    "freetime"
]
df=pd.read_csv(file_path)
df = df[cols]

# rename
df.rename(columns={"studytime": "study_hours"}, inplace=True)

# attendance (important logic)
df['attendance'] = 100 - df['absences']

# previous marks
df['previous_marks'] = (df['G1'] + df['G2']) / 2

# parental education
df['parental_education'] = (df['Medu'] + df['Fedu']) / 2
df['sleep_hours'] = np.random.randint(5, 9, size=len(df))
df['assignments_completed'] = np.random.randint(1, 10, size=len(df))
df['result'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
df.drop(["absences", "G1", "G2", "Medu", "Fedu"], axis=1, inplace=True)


# check the data have mussing value or not
# print(df.isnull().sum())
# print(df['result'].value_counts())

num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df.groupby('result')['G3'].describe()
df_model = df.drop(columns=['G3'])

X = df_model.drop(columns=['result'])
y = df_model['result']

print(X.columns.tolist())
num_cols = ['study_hours','freetime','attendance','previous_marks',
            'parental_education','sleep_hours','assignments_completed','mobile_usage_hours']

df_model['internet'] = df_model['internet'].map({'yes':1, 'no':0})

# 3. Feature engineering
df_model['study_per_freetime'] = df_model['study_hours'] / (df_model['freetime'] + 1)
df_model['effort_score'] = df_model['study_hours'] + df_model['assignments_completed'] + df_model['attendance']/20
df_model['marks_study_interaction'] = df_model['previous_marks'] * df_model['study_hours']

df_model['attendance_category'] = pd.cut(df_model['attendance'], bins=[0,75,90,100],
                                           labels=['low','medium','high'])
df_model = pd.get_dummies(df_model, columns=['attendance_category'], drop_first=True)

X = df_model.drop(columns=['result'])
y = df_model['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Baseline
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred = log_model.predict(X_test_scaled)

# RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)  # tree models scaling ki zaroorat nahi

y_pred_rf = rf_model.predict(X_test)

# create path
model_dir = os.path.join("..", "models")
# save models
# joblib.dump(log_model, os.path.join(model_dir, "logistic_model.pkl"))
# joblib.dump(rf_model, os.path.join(model_dir, "rf_model.pkl"))
# joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))


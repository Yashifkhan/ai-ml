import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

sample_student = pd.DataFrame([{
    'school': 'GP',
    'sex': 'F',
    'age': 16,
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'T',
    'Medu': 3,
    'Fedu': 2,
    'Mjob': 'teacher',
    'Fjob': 'other',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 1,
    'studytime': 3,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'no',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'no',
    'famrel': 4,
    'freetime': 3,
    'health': 4,
    'absences': 2,
    'sleep_hours': 7.0,
    'mobile_social_hours': 3.0,
    'Math_self_rating': 5,
    'Science_self _rating': 4,
    'Social_Science_self_rating': 3,
    'English_self_rating': 4,
    'Hindi_self_rating': 3
}])

print(sample_student.shape)
print(sample_student.columns.tolist())

sample_transformed = preprocessor.transform(sample_student)
prediction = model.predict(sample_transformed)

print("Predicted Average Marks:", prediction[0])

import joblib
import pandas as pd
linear_regression_model = joblib.load('linear_regression_pipeline.pkl')
decision_tree_model = joblib.load('decision_tree_pipeline.pkl')
random_forest_model = joblib.load('random_forest_pipeline.pkl')

# Step 2: Naya student data banao (raw form, jaisa pehle banaya tha)
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
    'Science_self_rating': 4,
    'Social_Science_self_rating': 3,
    'English_self_rating': 4,
    'Hindi_self_rating': 3
}])

# Step 3: Predict karo
prediction_lr = linear_regression_model.predict(sample_student)
prediction_dt = decision_tree_model.predict(sample_student)
prediction_rf = random_forest_model.predict(sample_student)

print("Predicted Average Marks Lr :", prediction_lr[0])
print("Predicted Average Marks Dt :", prediction_dt[0])
print("Predicted Average Marks Rf :", prediction_rf[0])
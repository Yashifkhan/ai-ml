import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load models
log_model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_model.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "models", "rf_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_marks: float
    parental_education: float
    sleep_hours: float
    assignments_completed: float
    mobile_usage_hours: float
    


@app.get("/model-test")
def test():
    return {"message":"Student pass fail"}


@app.post("/predict/logistic")
def predict_logistic(data: StudentData):
    input_data = [[
        data.study_hours,
        data.attendance,
        data.previous_marks,
        data.parental_education,
        data.sleep_hours,
        data.assignments_completed,
        data.mobile_usage_hours
    ]]

    scaled_data = scaler.transform(input_data)
    prediction = log_model.predict(scaled_data)

    return {"prediction": int(prediction[0])}


@app.post("/predict/rf")
def predict_rf(data: StudentData):
    input_data = [[
        data.study_hours,
        data.attendance,
        data.previous_marks,
        data.parental_education,
        data.sleep_hours,
        data.assignments_completed,
        data.mobile_usage_hours
    ]]

    prediction = rf_model.predict(input_data)

    return {"prediction": int(prediction[0])}
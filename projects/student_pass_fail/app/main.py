import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load models
log_model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_model.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "models", "rf_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

class StudentInput(BaseModel):
    study_hours: float
    internet: str
    freetime: float
    attendance: float
    previous_marks: float
    parental_education: float
    sleep_hours: float
    assignments_completed: float
    study_per_freetime: float
    effort_score: float
    marks_study_interaction: float
    attendance_category_medium: int
    attendance_category_high: int

@app.get("/model-test")
def test():
    return {"message":"Student pass fail api work"}


FEATURE_COLUMNS = [
    "study_hours",
    "internet",
    "freetime",
    "attendance",
    "previous_marks",
    "parental_education",
    "sleep_hours",
    "assignments_completed",
    "study_per_freetime",
    "effort_score",
    "marks_study_interaction",
    "attendance_category_medium",
    "attendance_category_high",
]

def build_feature_frame(data: StudentInput):
    payload = {
        "study_hours": data.study_hours,
        "internet": 1 if str(data.internet).lower() == "yes" else 0,
        "freetime": data.freetime,
        "attendance": data.attendance,
        "previous_marks": data.previous_marks,
        "parental_education": data.parental_education,
        "sleep_hours": data.sleep_hours,
        "assignments_completed": data.assignments_completed,
        "study_per_freetime": data.study_per_freetime,
        "effort_score": data.effort_score,
        "marks_study_interaction": data.marks_study_interaction,
        "attendance_category_medium": data.attendance_category_medium,
        "attendance_category_high": data.attendance_category_high,
    }
    return pd.DataFrame([payload], columns=FEATURE_COLUMNS)

@app.post("/predict/logistic")
def predict_logistic(data: StudentInput):
    df = build_feature_frame(data)

    scaled_data = scaler.transform(df[FEATURE_COLUMNS])
    prediction = log_model.predict(scaled_data)

    return {"prediction": int(prediction[0])}

@app.post("/predict/rf")
def predict_rf(data: StudentInput):
    payload = {
        "study_hours": data.study_hours,
        "internet": 1 if data.internet.lower() == "yes" else 0,
        "freetime": data.freetime,
        "attendance": data.attendance,
        "previous_marks": data.previous_marks,
        "parental_education": data.parental_education,
        "sleep_hours": data.sleep_hours,
        "assignments_completed": data.assignments_completed,
        "study_per_freetime": data.study_per_freetime,
        "effort_score": data.effort_score,
        "marks_study_interaction": data.marks_study_interaction,
        "attendance_category_medium": data.attendance_category_medium,
        "attendance_category_high": data.attendance_category_high,
    }

    df = pd.DataFrame([payload], columns=[
        "study_hours", "internet", "freetime", "attendance", "previous_marks",
        "parental_education", "sleep_hours", "assignments_completed",
        "study_per_freetime", "effort_score", "marks_study_interaction",
        "attendance_category_medium", "attendance_category_high"
    ])

    pred = rf_model.predict(df)
    return {"prediction": int(pred[0])}
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import DeveloperProfile
from app.model_loader import load_model   # 👈 direct import
from app.pipeline import DeveloperSalaryPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()   # 👈 yahan explicitly call kar rahe hain

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running 🚀"}


@app.get("/test-model")
def test_model():
    return {"models_loaded": list(model.keys())}


@app.post("/predict-emp-salary")
def get_prediction(profile: DeveloperProfile):
    import pandas as pd
    input_df = pd.DataFrame([profile.dict()])
    prediction = model.predict(input_df)
    return {"predicted_salary": round(float(prediction[0]), 2)}
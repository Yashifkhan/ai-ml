# app/predict.py
import pandas as pd
from app.model_loader import model
from app.schemas import DeveloperProfile

def predict_salary(profile: DeveloperProfile):
    input_df = pd.DataFrame([profile.dict()])
    prediction = model.predict(input_df)
    return float(prediction[0])
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd

from app.predict import predict_churn

app = FastAPI(title="Customer Churn Prediction API")

# CORS — React frontend se calls allow karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production me apne frontend ka exact URL daalo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------ REQUEST SCHEMA ------------------
# Raw form fields jo React se aayenge (get_dummies se pehle ka original data)

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float
    model_choice: Optional[Literal["random_forest", "logistic_regression"]] = "random_forest"


# ------------------ PREPROCESSING (same logic as training) ------------------

def preprocess_input(data: CustomerData) -> pd.DataFrame:
    row = data.dict()
    model_choice = row.pop("model_choice")

    df = pd.DataFrame([row])

    # binary mappings — training ke exact same encoding
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    binary_yes_no_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # feature engineering — training jaisa hi
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
    )

    # one-hot encoding — same columns jo training me the
    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod", "InternetService", "tenure_group"],
        drop_first=True
    )

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df, model_choice


# ------------------ ROUTES ------------------

@app.get("/model-test")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/costumer-churn-predict")
def predict(data: CustomerData):
    try:
        processed_df, model_choice = preprocess_input(data)
        result = predict_churn(processed_df, model_choice=model_choice)
        return {
            "model_used": model_choice,
            "churn_prediction": "Yes" if result["churn_prediction"] == 1 else "No",
            "churn_probability": result["probability"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback # Add this to the top of your imports

# CRITICAL: We must import the custom class from your pipeline file so joblib can rebuild it
# from train_model import DeveloperSalaryPipeline
from pipeline import DeveloperSalaryPipeline

app = FastAPI(title="Developer Salary Predictor API")

# Allow the frontend (HTML file) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for local development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We load the model once when the server starts, NOT every time a request is made.
# This keeps the API extremely fast.


class DeveloperProfile(BaseModel):
    """
    Pydantic model to enforce data types and structure coming from the frontend.
    These exact keys match what the XGBoost pipeline expects.
    """
    YearsCodePro: str
    EdLevel: str
    DevType: str
    OrgSize: str
    Industry: str
    RemoteWork: str
    Country: str
    LanguageHaveWorkedWith: str
    PlatformHaveWorkedWith: str
    DatabaseHaveWorkedWith: str
    ToolsTechHaveWorkedWith: str

@app.post("/predict")
async def predict_salary(profile: DeveloperProfile):
    """
    Receives JSON data from the frontend, converts it to a Pandas DataFrame,
    runs it through the pipeline, and returns the predicted salary.
    """
    # Great for debugging - shows exactly what the React app sent
    print(f"Incoming Profile Data: {profile}") 
    model = joblib.load("developer_salary_model.pkl")
    
    
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    try:
        # 1. Convert the incoming Pydantic object to a dictionary
        # Use .model_dump() for Pydantic v2 (FastAPI standard), fallback to .dict() for v1
        input_dict = profile.model_dump() if hasattr(profile, 'model_dump') else profile.dict()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # 2. Run the prediction using the custom pipeline
        print("Input DataFrame:")
        print(input_df)
        print("Columns:", input_df.columns.tolist())
        prediction = model.predict(input_df)
        
        # 3. Format the result and return it
        predicted_salary = float(prediction[0])
        
        # Optional: Ensure the model doesn't predict negative salaries for extreme edge cases
        predicted_salary = max(0.0, predicted_salary)
        
        return {
            "status": "success",
            "predicted_salary_usd": predicted_salary,
            "formatted_salary": f"${predicted_salary:,.2f}"
        }
        
    except Exception as e:
        # Print the full error traceback to your terminal for easier debugging
        print("❌ Pipeline Error:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Salary Predictor API is running! Send a POST request to /predict."}

# To run this server, open your terminal and type:
# uvicorn main:app --reload
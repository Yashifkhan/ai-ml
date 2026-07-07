from fastapi import FastAPI
from app.model_loader import load_models
from app.predict import predict_student
from app.schemas import StudentData
from fastapi.middleware.cors import CORSMiddleware

# create the app 
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# load the ,models 
models=load_models()
@app.get("/")
def home():
    return {"message": "Student Mark predicter api App run"}


@app.get("/test-model")
def test_model():
    return {"models_loaded": list(models.keys())}

@app.post("/predict")
def predict(data: StudentData):
    result = predict_student(models, data)
    return {"prediction": result}
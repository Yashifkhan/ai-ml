from app.pipeline import DeveloperSalaryPipeline

MODEL_PATH = "app/developer_salary_model.pkl"   # 👈 same path jaha save hua

def load_model():
    return DeveloperSalaryPipeline.load(MODEL_PATH)

model = load_model()
import pandas as pd

def predict_student(models, data):
    
    df = pd.DataFrame([data.dict()])   # convert input → dataframe

    return {
        "linear_regression": models["lr"].predict(df)[0],
        "decision_tree": models["dt"].predict(df)[0],
        "random_forest": models["rf"].predict(df)[0]
    }
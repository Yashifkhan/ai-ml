# import pandas as pd

# def predict_churn(input_df, model_choice="random_forest"):
#     log_model, rf_model, scaler, feature_columns, thresholds = load_artifacts()

#     # ensure same columns as training (missing columns add karo with 0)
#     input_df = input_df.reindex(columns=feature_columns, fill_value=0)

#     if model_choice == "random_forest":
#         prob = rf_model.predict_proba(input_df)[:, 1][0]
#         threshold = thresholds["random_forest"]
#     else:
#         input_scaled = scaler.transform(input_df)
#         prob = log_model.predict_proba(input_scaled)[:, 1][0]
#         threshold = thresholds["logistic_regression"]

#     prediction = int(prob >= threshold)
#     return {"churn_prediction": prediction, "probability": round(float(prob), 4)}


import pandas as pd
from app.model_loader import load_artifacts


def predict_churn(input_df, model_choice="random_forest"):
    log_model, rf_model, scaler, feature_columns, thresholds = load_artifacts()

    # ensure same columns as training (missing columns add karo with 0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    if model_choice == "random_forest":
        prob = rf_model.predict_proba(input_df)[:, 1][0]
        threshold = thresholds["random_forest"]
    else:
        input_scaled = scaler.transform(input_df)
        prob = log_model.predict_proba(input_scaled)[:, 1][0]
        threshold = thresholds["logistic_regression"]

    prediction = int(prob >= threshold)
    return {"churn_prediction": prediction, "probability": round(float(prob), 4)}
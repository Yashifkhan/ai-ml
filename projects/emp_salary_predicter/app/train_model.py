# import pandas as pd
# import numpy as np
# import re
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import RandomizedSearchCV


# required_features=[
#          "YearsCodePro",
#          "WorkExp",
#          "EdLevel",
#          "DevType",
#          "OrgSize",
#          "Industry",
#          "RemoteWork",
#          "Country",
#          "LanguageHaveWorkedWith",
#          "PlatformHaveWorkedWith",
#          "DatabaseHaveWorkedWith",
#          "ToolsTechHaveWorkedWith",
#          "ConvertedCompYearly",
# ]


# df=pd.read_csv("../data/survey_results_public.csv",usecols=required_features)

# # data understanding and 
# # print the all uniqe skills user select in which 
# # skills_column = df["LanguageHaveWorkedWith"]
# # skills_column = skills_column.dropna()
# # all_skills = skills_column.str.split(";")
# # flat_skills = [skill for sublist in all_skills for skill in sublist]
# # unique_skills = set(flat_skills)
# # print("Total unique skills:", len(unique_skills))
# # print(unique_skills)

# # print the all uniqe db 
# # db_skills = df["DatabaseHaveWorkedWith"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)


# # print the all uniqe PlatformHaveWorkedWith 
# # db_skills = df["PlatformHaveWorkedWith"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)




# === Grandmaster Pipeline: Stacked Models + Strict Bounds ===
# MAE:  $23017.02
# RMSE: $32516.72
# R² Score: 0.6311

# app/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from app.pipeline import DeveloperSalaryPipeline   # 👈 import from pipeline.py

if __name__ == "__main__":
    required_features = [
        "YearsCodePro", "EdLevel", "DevType", "OrgSize", "Industry",
        "RemoteWork", "Country", "LanguageHaveWorkedWith", "PlatformHaveWorkedWith",
        "DatabaseHaveWorkedWith", "ToolsTechHaveWorkedWith", "ConvertedCompYearly"
    ]
    df = pd.read_csv("data/survey_results_public.csv", usecols=required_features)
    df = df[df['ConvertedCompYearly'].notnull()].copy()
    df = df[df['LanguageHaveWorkedWith'].notnull()].copy()
    df = df[df['DevType'].notnull()].copy()
    df = df[df['Country'].notnull()].copy()

    top_countries = df['Country'].value_counts().head(15).index.tolist()
    df['Temp_Country_Group'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')

    def filter_local_outliers(group):
        Q1 = group['ConvertedCompYearly'].quantile(0.15)
        Q3 = group['ConvertedCompYearly'].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR
        return group[(group['ConvertedCompYearly'] >= lower_bound) & (group['ConvertedCompYearly'] <= upper_bound)]

    df = df.groupby('Temp_Country_Group', group_keys=False).apply(filter_local_outliers).reset_index(drop=True)
    df = df.drop(columns=['Temp_Country_Group'])

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    pipeline = DeveloperSalaryPipeline()
    pipeline.fit(df_train)
    pipeline.save("app/developer_salary_model.pkl")   # apne actual path se adjust karo








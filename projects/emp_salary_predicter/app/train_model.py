

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








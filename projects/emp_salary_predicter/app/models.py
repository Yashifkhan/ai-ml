import pandas as pd
import numpy as np

df=pd.read_csv("../data/survey_results_public.csv")
# get the requred feature 
required_features=[
         "YearsCodePro",
         "WorkExp",
         "EdLevel",
         "DevType",
         "OrgSize",
         "Industry",
         "RemoteWork",
         "Country",
         "LanguageHaveWorkedWith",
         "PlatformHaveWorkedWith",
         "DatabaseHaveWorkedWith",
         "ToolsTechHaveWorkedWith",
         "ConvertedCompYearly",

]

df_selected=df[required_features]

# feature engineering 
# step 1 

# target column is missing many rows so we drop this 
df_step2 = df_selected.dropna(subset=["ConvertedCompYearly"])
# print("Shape after dropping missing target:", df_step2.shape)
# print("\nTarget statistics:\n", df_step2["ConvertedCompYearly"].describe())
# print("\nTop 10 highest values:\n", df_step2["ConvertedCompYearly"].sort_values(ascending=False).head(10))
# print("\nBottom 10 lowest values:\n", df_step2["ConvertedCompYearly"].sort_values(ascending=True).head(10))


print("1st percentile:", df_step2["ConvertedCompYearly"].quantile(0.01))
print("99th percentile:", df_step2["ConvertedCompYearly"].quantile(0.99))

# IQR method boundaries
Q1 = df_step2["ConvertedCompYearly"].quantile(0.25)
Q3 = df_step2["ConvertedCompYearly"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
lower_cutoff = df_step2["ConvertedCompYearly"].quantile(0.01)
upper_cutoff = df_step2["ConvertedCompYearly"].quantile(0.99)

df_step2_clean = df_step2[
    (df_step2["ConvertedCompYearly"] >= lower_cutoff) &
    (df_step2["ConvertedCompYearly"] <= upper_cutoff)
]

def convert_years(val):
    if pd.isna(val):
        return np.nan
    if val == "Less than 1 year":
        return 0
    if val == "More than 50 years":
        return 51
    return int(val)

non_numeric = df_step2_clean["YearsCodePro"][~df_step2_clean["YearsCodePro"].str.isnumeric().fillna(False)]
df_step2_clean["YearsCodePro"] = df_step2_clean["YearsCodePro"].apply(convert_years)

# Step A: Fill missing YearsCodePro using WorkExp (reverse direction) where possible
df_step2_clean["YearsCodePro"] = df_step2_clean["YearsCodePro"].fillna(df_step2_clean["WorkExp"])

# Step B: Fill missing WorkExp using YearsCodePro
df_step2_clean["WorkExp"] = df_step2_clean["WorkExp"].fillna(df_step2_clean["YearsCodePro"])

# Step C: Whatever is STILL missing in either (cases where both were missing) -> fill with median
median_years = df_step2_clean["YearsCodePro"].median()
df_step2_clean["YearsCodePro"] = df_step2_clean["YearsCodePro"].fillna(median_years)
df_step2_clean["WorkExp"] = df_step2_clean["WorkExp"].fillna(median_years)

# Verify
categorical_cols = ["EdLevel", "DevType", "OrgSize", "Industry", "RemoteWork", "Country"]

edlevel_order = {
    "Primary/elementary school": 0,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 1,
    "Some college/university study without earning a degree": 2,
    "Associate degree (A.A., A.S., etc.)": 3,
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 4,
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 6,
    "Something else": 3,   # neutral/unknown -> mid value
    "Unknown": 3            # missing -> mid value
}

orgsize_order = {
    "Just me - I am a freelancer, sole proprietor, etc.": 0,
    "2 to 9 employees": 1,
    "10 to 19 employees": 2,
    "20 to 99 employees": 3,
    "100 to 499 employees": 4,
    "500 to 999 employees": 5,
    "1,000 to 4,999 employees": 6,
    "5,000 to 9,999 employees": 7,
    "10,000 or more employees": 8,
    "I don’t know": 4,      # unknown -> mid value
    "Unknown": 4             # missing -> mid value
}

df_step2_clean["EdLevel_encoded"] = df_step2_clean["EdLevel"].map(edlevel_order)
df_step2_clean["OrgSize_encoded"] = df_step2_clean["OrgSize"].map(orgsize_order)

df_step2_clean = df_step2_clean.copy()

df_step2_clean = pd.get_dummies(df_step2_clean, columns=["RemoteWork", "Industry"], prefix=["Remote", "Industry"])

print("New shape:", df_step2_clean.shape)
print("\nNew columns added:\n", [col for col in df_step2_clean.columns if col.startswith("Remote_") or col.startswith("Industry_")])


# print("\nSelected shape:", df_selected.shape)
# print("\nDtypes:\n", df_selected.dtypes)
# print("\nMissing value % per column:\n", (df_selected.isnull().mean()*100).round(2))
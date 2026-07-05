import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV


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


df=pd.read_csv("../data/survey_results_public.csv",usecols=required_features)

# data understanding and 
# print the all uniqe skills user select in which 
# skills_column = df["LanguageHaveWorkedWith"]
# skills_column = skills_column.dropna()
# all_skills = skills_column.str.split(";")
# flat_skills = [skill for sublist in all_skills for skill in sublist]
# unique_skills = set(flat_skills)
# print("Total unique skills:", len(unique_skills))
# print(unique_skills)

# print the all uniqe db 
# db_skills = df["DatabaseHaveWorkedWith"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# print the all uniqe PlatformHaveWorkedWith 
# db_skills = df["PlatformHaveWorkedWith"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# print all the unieq ToolsTechHaveWorkedWith
# db_skills = df["ToolsTechHaveWorkedWith"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# print all the unieq ToolsTechHaveWorkedWith
# db_skills = df["EdLevel"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)

# db_skills = df["DevType"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)

# db_skills = df["YearsCodePro"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# db_skills = df["OrgSize"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)
# print("data filter proper for the ui filters ")


# eda perform on data and create for model 
df = df[df['ConvertedCompYearly'].notnull()].copy()


lower = df['ConvertedCompYearly'].quantile(0.01)
upper = df['ConvertedCompYearly'].quantile(0.99)

df = df[(df['ConvertedCompYearly'] >= lower) & (df['ConvertedCompYearly'] <= upper)].copy()



df['LogConvertedCompYearly'] = np.log1p(df['ConvertedCompYearly'])
df = df.drop(columns=['LogConvertedCompYearly'])

df['YearsCodePro'] = df['YearsCodePro'].replace({
    'Less than 1 year': 0,
    'More than 50 years': 51
})

# Step 2: Convert to numeric
df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'], errors='coerce')

# Step 3: Verify
median_years = df['YearsCodePro'].median()

df['YearsCodePro'] = df['YearsCodePro'].fillna(median_years)

# Verify
correlation = df[['YearsCodePro', 'WorkExp']].corr()
# print(correlation)

# Also compare their distributions side by side
df = df.drop(columns=['WorkExp'])

edlevel_mapping = {
    "Primary/elementary school": "Primary",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary",
    "Some college/university study without earning a degree": "Some College",
    "Associate degree (A.A., A.S., etc.)": "Associate",
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional/PhD",
    "Something else": "Other"
}

df['EdLevel'] = df['EdLevel'].map(edlevel_mapping)

# Verify
# Drop rows with missing RemoteWork
df = df[df['RemoteWork'].notnull()].copy()

df = df[df['OrgSize'].notnull()].copy()

orgsize_mapping = {
    "Just me - I am a freelancer, sole proprietor, etc.": 0,
    "2 to 9 employees": 1,
    "10 to 19 employees": 2,
    "20 to 99 employees": 3,
    "100 to 499 employees": 4,
    "500 to 999 employees": 5,
    "1,000 to 4,999 employees": 6,
    "5,000 to 9,999 employees": 7,
    "10,000 or more employees": 8,
    "I don’t know": -1   # separate bucket, not part of the order
}

df['OrgSize_encoded'] = df['OrgSize'].map(orgsize_mapping)

# Verify
df = df[df['DevType'].notnull()].copy()



# Top 20 countries by count
top_countries = df['Country'].value_counts().head(15).index.tolist()

# Group everything else into "Other"
df['Country_grouped'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')

# Verify
df['Industry'] = df['Industry'].fillna('Not Specified')

# Verify
df = df[df['LanguageHaveWorkedWith'].notnull()].copy()

# Split by semicolon and count frequency of each language
from collections import Counter

all_languages = df['LanguageHaveWorkedWith'].str.split(';').explode()
lang_counts = Counter(all_languages)

## Get ALL unique languages
all_lang_list = list(lang_counts.keys())

# Create binary flag columns for each language
for lang in all_lang_list:
    col_name = f"lang_{lang.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '')}"
    df[col_name] = df['LanguageHaveWorkedWith'].apply(
        lambda x: 1 if lang in x.split(';') else 0
    )

# Verify

from collections import Counter
all_dbs = df['DatabaseHaveWorkedWith'].dropna().str.split(';').explode()
db_counts = Counter(all_dbs)
db_series = pd.Series(db_counts).sort_values(ascending=False)
top_databases = db_series.head(10).index.tolist()

# Create binary flags - missing values automatically become 0 (no database flagged)
for db in top_databases:
    col_name = f"db_{db.replace(' ', '_').replace('.', '')}"
    df[col_name] = df['DatabaseHaveWorkedWith'].apply(
        lambda x: 1 if pd.notnull(x) and db in x.split(';') else 0
    )

# Verify
new_db_cols = [c for c in df.columns if c.startswith('db_')]

# Count unique platforms
all_platforms = df['PlatformHaveWorkedWith'].dropna().str.split(';').explode()
platform_counts = Counter(all_platforms)
platform_series = pd.Series(platform_counts).sort_values(ascending=False)
top_platforms = platform_series.head(8).index.tolist()
# print("Top 8 platforms:", top_platforms)

# Create binary flags - missing values become 0 automatically
for platform in top_platforms:
    col_name = f"platform_{platform.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
    df[col_name] = df['PlatformHaveWorkedWith'].apply(
        lambda x: 1 if pd.notnull(x) and platform in x.split(';') else 0
    )

# Verify
new_platform_cols = [c for c in df.columns if c.startswith('platform_')]


# Count unique tools
all_tools = df['ToolsTechHaveWorkedWith'].dropna().str.split(';').explode()
tool_counts = Counter(all_tools)
tool_series = pd.Series(tool_counts).sort_values(ascending=False)
top_tools = tool_series.head(12).index.tolist()

# Create binary flags - missing values become 0 automatically
for tool in top_tools:
    col_name = f"tool_{tool.replace(' ', '_').replace('(', '').replace(')', '').replace(chr(39), '')}"
    df[col_name] = df['ToolsTechHaveWorkedWith'].apply(
        lambda x: 1 if pd.notnull(x) and tool in x.split(';') else 0
    )

# Verify
new_tool_cols = [c for c in df.columns if c.startswith('tool_')]
columns_to_drop = [
    'Country',                    # replaced by Country_grouped
    'OrgSize',                    # replaced by OrgSize_encoded
    'LanguageHaveWorkedWith',     # replaced by lang_* flags
    'DatabaseHaveWorkedWith',     # replaced by db_* flags
    'PlatformHaveWorkedWith',     # replaced by platform_* flags
    'ToolsTechHaveWorkedWith',    # replaced by tool_* flags
]

df_final = df.drop(columns=columns_to_drop)

#  One-hot encode remaining categorical columns
categorical_cols = ['RemoteWork', 'EdLevel', 'DevType', 'Industry', 'Country_grouped']

# One-hot encode
df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)

from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df_final.drop(columns=['ConvertedCompYearly'])
y = df_final['ConvertedCompYearly']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print("\nTrain shape:", X_train.shape)
# print("Test shape:", X_test.shape)


# test the data to ready for the model or not 
# print("Missing in X_train:", X_train.isnull().sum().sum())
# print("Missing in X_test:", X_test.isnull().sum().sum())
# print("Missing in y_train:", y_train.isnull().sum())
# print("Missing in y_test:", y_test.isnull().sum())
# print("Infinite in X_train:", np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum())
# print("Duplicate rows in X:", X.duplicated().sum())
# print(X_train.describe().T[['min', 'max', 'mean', 'std']])
# print("y_train stats:\n", y_train.describe())
# print("\ny_test stats:\n", y_test.describe())
# print(X.columns.tolist())

# Check duplicates in the full X (before split)

# Clean column names - replace special characters with underscore
# Step 1: Rebuild X_clean fresh from the ORIGINAL X (with original column names intact)
duplicate_mask = X.duplicated()
X_clean = X[~duplicate_mask].reset_index(drop=True)
y_clean = y[~duplicate_mask].reset_index(drop=True)


# Step 2: Manual rename FIRST, before any regex touches these columns
manual_renames = {
    'lang_C#': 'lang_Csharp',
    'lang_C++': 'lang_Cpp',
    'lang_F#': 'lang_Fsharp',
}
X_clean = X_clean.rename(columns=manual_renames)

# Step 3: Now apply generic regex cleaning to everything else
X_clean.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X_clean.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# train a model 
# Train baseline Random Forest
# model = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)
# # Predict
# y_pred = model.predict(X_test)
# # Evaluate
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print(f"MAE: {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.4f}")

# Create interaction feature: US-based AND senior role
senior_roles = ['DevType_Senior_Executive_C_Suite_VP_etc_', 'DevType_Engineering_manager']

X_train_new = X_train.copy()
X_test_new = X_test.copy()

X_train_new['US_and_Senior'] = X_train['Country_grouped_United_States_of_America'] * X_train[senior_roles].max(axis=1)
X_test_new['US_and_Senior'] = X_test['Country_grouped_United_States_of_America'] * X_test[senior_roles].max(axis=1)

# Retrain with the new interaction feature
# model_v2 = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )

# model_v2.fit(X_train_new, y_train)
# y_pred_v2 = model_v2.predict(X_test_new)

# # Evaluate
# mae_v2 = mean_absolute_error(y_test, y_pred_v2)
# rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_v2))
# r2_v2 = r2_score(y_test, y_pred_v2)

# print("=== Before (baseline) ===")
# print(f"MAE: 28641.49, RMSE: 41922.01, R²: 0.5880")

# print("\n=== After (with US_and_Senior feature) ===")
# print(f"MAE: {mae_v2:.2f}")
# print(f"RMSE: {rmse_v2:.2f}")
# print(f"R² Score: {r2_v2:.4f}")

# # Check if the new feature made it into top importances
# importances_v2 = pd.DataFrame({
#     'feature': X_train_new.columns,
#     'importance': model_v2.feature_importances_
# }).sort_values('importance', ascending=False)

# print("\nWhere does US_and_Senior rank?")
# print(importances_v2[importances_v2['feature'] == 'US_and_Senior'])
# print("\nTop 10 features now:")
# print(importances_v2.head(10).to_string(index=False))


# Train XGBoost with reasonable starting params
# model_xgb = xgb.XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )

# model_xgb.fit(X_train_new, y_train)
# y_pred_xgb = model_xgb.predict(X_test_new)

# Evaluate
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
# rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# r2_xgb = r2_score(y_test, y_pred_xgb)

# residuals_xgb = y_test - y_pred_xgb
# abs_residuals_xgb = np.abs(residuals_xgb)

# error_df_xgb = pd.DataFrame({
#     'actual': y_test.values,
#     'predicted': y_pred_xgb,
#     'error': residuals_xgb.values,
#     'abs_error': abs_residuals_xgb.values
# }).sort_values('abs_error', ascending=False)

#
model_v3 = xgb.XGBRegressor(
    subsample=0.6,
    reg_lambda=1.5,
    reg_alpha=0,
    n_estimators=500,
    min_child_weight=3,
    max_depth=10,
    learning_rate=0.03,
    colsample_bytree=0.6,
    random_state=42,
    n_jobs=-1
)

model_v3.fit(X_train_new, y_train)
y_pred_v3 = model_v3.predict(X_test_new)

mae_v3 = mean_absolute_error(y_test, y_pred_v3)
rmse_v3 = np.sqrt(mean_squared_error(y_test, y_pred_v3))
r2_v3 = r2_score(y_test, y_pred_v3)

print("=== Previous best (tuned XGBoost) ===")
print("MAE: 27120.50, RMSE: 40078.80, R²: 0.6235")

print("\n=== With new features (skill_count + experience_bucket) ===")
print(f"MAE: {mae_v3:.2f}")
print(f"RMSE: {rmse_v3:.2f}")
print(f"R² Score: {r2_v3:.4f}")

# Check where new features rank in importance
importances_v3 = pd.DataFrame({
    'feature': X_train_new.columns,
    'importance': model_v3.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features now:")
print(importances_v3.head(15).to_string(index=False))
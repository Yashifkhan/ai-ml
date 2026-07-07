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


# # print all the unieq ToolsTechHaveWorkedWith
# # db_skills = df["ToolsTechHaveWorkedWith"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)


# # print all the unieq ToolsTechHaveWorkedWith
# # db_skills = df["EdLevel"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)

# # db_skills = df["DevType"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)

# # db_skills = df["YearsCodePro"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)


# # db_skills = df["OrgSize"]
# # db_column = db_skills.dropna()
# # all_db = db_column.str.split(";")
# # flat_skills = [skill for sublist in all_db for skill in sublist]
# # unique_db = set(flat_skills)
# # print("Total unique db  skills:", len(unique_db))
# # print(unique_db)
# # print("data filter proper for the ui filters ")


# # eda perform on data and create for model 
# df = df[df['ConvertedCompYearly'].notnull()].copy()


# lower = df['ConvertedCompYearly'].quantile(0.01)
# upper = df['ConvertedCompYearly'].quantile(0.99)

# df = df[(df['ConvertedCompYearly'] >= lower) & (df['ConvertedCompYearly'] <= upper)].copy()



# df['LogConvertedCompYearly'] = np.log1p(df['ConvertedCompYearly'])
# df = df.drop(columns=['LogConvertedCompYearly'])

# df['YearsCodePro'] = df['YearsCodePro'].replace({
#     'Less than 1 year': 0,
#     'More than 50 years': 51
# })

# # Step 2: Convert to numeric
# df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'], errors='coerce')

# # Step 3: Verify
# median_years = df['YearsCodePro'].median()

# df['YearsCodePro'] = df['YearsCodePro'].fillna(median_years)

# # Verify
# correlation = df[['YearsCodePro', 'WorkExp']].corr()
# # print(correlation)

# # Also compare their distributions side by side
# df = df.drop(columns=['WorkExp'])

# edlevel_mapping = {
#     "Primary/elementary school": "Primary",
#     "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary",
#     "Some college/university study without earning a degree": "Some College",
#     "Associate degree (A.A., A.S., etc.)": "Associate",
#     "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
#     "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
#     "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional/PhD",
#     "Something else": "Other"
# }

# df['EdLevel'] = df['EdLevel'].map(edlevel_mapping)

# # Verify
# # Drop rows with missing RemoteWork
# df = df[df['RemoteWork'].notnull()].copy()

# df = df[df['OrgSize'].notnull()].copy()

# orgsize_mapping = {
#     "Just me - I am a freelancer, sole proprietor, etc.": 0,
#     "2 to 9 employees": 1,
#     "10 to 19 employees": 2,
#     "20 to 99 employees": 3,
#     "100 to 499 employees": 4,
#     "500 to 999 employees": 5,
#     "1,000 to 4,999 employees": 6,
#     "5,000 to 9,999 employees": 7,
#     "10,000 or more employees": 8,
#     "I don’t know": -1   # separate bucket, not part of the order
# }

# df['OrgSize_encoded'] = df['OrgSize'].map(orgsize_mapping)

# # Verify
# df = df[df['DevType'].notnull()].copy()



# # Top 20 countries by count
# top_countries = df['Country'].value_counts().head(15).index.tolist()

# # Group everything else into "Other"
# df['Country_grouped'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')

# # Verify
# df['Industry'] = df['Industry'].fillna('Not Specified')

# # Verify
# df = df[df['LanguageHaveWorkedWith'].notnull()].copy()

# # Split by semicolon and count frequency of each language
# from collections import Counter

# all_languages = df['LanguageHaveWorkedWith'].str.split(';').explode()
# lang_counts = Counter(all_languages)

# ## Get ALL unique languages
# all_lang_list = list(lang_counts.keys())

# # Create binary flag columns for each language
# for lang in all_lang_list:
#     col_name = f"lang_{lang.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '')}"
#     df[col_name] = df['LanguageHaveWorkedWith'].apply(
#         lambda x: 1 if lang in x.split(';') else 0
#     )

# # Verify

# from collections import Counter
# all_dbs = df['DatabaseHaveWorkedWith'].dropna().str.split(';').explode()
# db_counts = Counter(all_dbs)
# db_series = pd.Series(db_counts).sort_values(ascending=False)
# top_databases = db_series.head(10).index.tolist()

# # Create binary flags - missing values automatically become 0 (no database flagged)
# for db in top_databases:
#     col_name = f"db_{db.replace(' ', '_').replace('.', '')}"
#     df[col_name] = df['DatabaseHaveWorkedWith'].apply(
#         lambda x: 1 if pd.notnull(x) and db in x.split(';') else 0
#     )

# # Verify
# new_db_cols = [c for c in df.columns if c.startswith('db_')]

# # Count unique platforms
# all_platforms = df['PlatformHaveWorkedWith'].dropna().str.split(';').explode()
# platform_counts = Counter(all_platforms)
# platform_series = pd.Series(platform_counts).sort_values(ascending=False)
# top_platforms = platform_series.head(8).index.tolist()
# # print("Top 8 platforms:", top_platforms)

# # Create binary flags - missing values become 0 automatically
# for platform in top_platforms:
#     col_name = f"platform_{platform.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
#     df[col_name] = df['PlatformHaveWorkedWith'].apply(
#         lambda x: 1 if pd.notnull(x) and platform in x.split(';') else 0
#     )

# # Verify
# new_platform_cols = [c for c in df.columns if c.startswith('platform_')]


# # Count unique tools
# all_tools = df['ToolsTechHaveWorkedWith'].dropna().str.split(';').explode()
# tool_counts = Counter(all_tools)
# tool_series = pd.Series(tool_counts).sort_values(ascending=False)
# top_tools = tool_series.head(12).index.tolist()

# # Create binary flags - missing values become 0 automatically
# for tool in top_tools:
#     col_name = f"tool_{tool.replace(' ', '_').replace('(', '').replace(')', '').replace(chr(39), '')}"
#     df[col_name] = df['ToolsTechHaveWorkedWith'].apply(
#         lambda x: 1 if pd.notnull(x) and tool in x.split(';') else 0
#     )

# # Verify
# new_tool_cols = [c for c in df.columns if c.startswith('tool_')]
# columns_to_drop = [
#     'Country',                    # replaced by Country_grouped
#     'OrgSize',                    # replaced by OrgSize_encoded
#     'LanguageHaveWorkedWith',     # replaced by lang_* flags
#     'DatabaseHaveWorkedWith',     # replaced by db_* flags
#     'PlatformHaveWorkedWith',     # replaced by platform_* flags
#     'ToolsTechHaveWorkedWith',    # replaced by tool_* flags
# ]

# df_final = df.drop(columns=columns_to_drop)

# #  One-hot encode remaining categorical columns
# categorical_cols = ['RemoteWork', 'EdLevel', 'DevType', 'Industry', 'Country_grouped']

# # One-hot encode
# df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)

# from sklearn.model_selection import train_test_split

# # Separate features (X) and target (y)
# X = df_final.drop(columns=['ConvertedCompYearly'])
# y = df_final['ConvertedCompYearly']

# # Train-test split (80-20)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # print("\nTrain shape:", X_train.shape)
# # print("Test shape:", X_test.shape)


# # test the data to ready for the model or not 
# # print("Missing in X_train:", X_train.isnull().sum().sum())
# # print("Missing in X_test:", X_test.isnull().sum().sum())
# # print("Missing in y_train:", y_train.isnull().sum())
# # print("Missing in y_test:", y_test.isnull().sum())
# # print("Infinite in X_train:", np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum())
# # print("Duplicate rows in X:", X.duplicated().sum())
# # print(X_train.describe().T[['min', 'max', 'mean', 'std']])
# # print("y_train stats:\n", y_train.describe())
# # print("\ny_test stats:\n", y_test.describe())
# # print(X.columns.tolist())

# # Check duplicates in the full X (before split)

# # Clean column names - replace special characters with underscore
# # Step 1: Rebuild X_clean fresh from the ORIGINAL X (with original column names intact)
# duplicate_mask = X.duplicated()
# X_clean = X[~duplicate_mask].reset_index(drop=True)
# y_clean = y[~duplicate_mask].reset_index(drop=True)


# # Step 2: Manual rename FIRST, before any regex touches these columns
# manual_renames = {
#     'lang_C#': 'lang_Csharp',
#     'lang_C++': 'lang_Cpp',
#     'lang_F#': 'lang_Fsharp',
# }
# X_clean = X_clean.rename(columns=manual_renames)

# # Step 3: Now apply generic regex cleaning to everything else
# X_clean.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X_clean.columns]

# X_train, X_test, y_train, y_test = train_test_split(
#     X_clean, y_clean, test_size=0.2, random_state=42
# )

# # train a model 
# # Train baseline Random Forest
# # model = RandomForestRegressor(
# #     n_estimators=100,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # model.fit(X_train, y_train)
# # # Predict
# # y_pred = model.predict(X_test)
# # # Evaluate
# # mae = mean_absolute_error(y_test, y_pred)
# # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # r2 = r2_score(y_test, y_pred)

# # print(f"MAE: {mae:.2f}")
# # print(f"RMSE: {rmse:.2f}")
# # print(f"R² Score: {r2:.4f}")

# # Create interaction feature: US-based AND senior role
# senior_roles = ['DevType_Senior_Executive_C_Suite_VP_etc_', 'DevType_Engineering_manager']

# X_train_new = X_train.copy()
# X_test_new = X_test.copy()

# X_train_new['US_and_Senior'] = X_train['Country_grouped_United_States_of_America'] * X_train[senior_roles].max(axis=1)
# X_test_new['US_and_Senior'] = X_test['Country_grouped_United_States_of_America'] * X_test[senior_roles].max(axis=1)

# # Retrain with the new interaction feature
# # model_v2 = RandomForestRegressor(
# #     n_estimators=100,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # model_v2.fit(X_train_new, y_train)
# # y_pred_v2 = model_v2.predict(X_test_new)

# # # Evaluate
# # mae_v2 = mean_absolute_error(y_test, y_pred_v2)
# # rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_v2))
# # r2_v2 = r2_score(y_test, y_pred_v2)

# # print("=== Before (baseline) ===")
# # print(f"MAE: 28641.49, RMSE: 41922.01, R²: 0.5880")

# # print("\n=== After (with US_and_Senior feature) ===")
# # print(f"MAE: {mae_v2:.2f}")
# # print(f"RMSE: {rmse_v2:.2f}")
# # print(f"R² Score: {r2_v2:.4f}")

# # # Check if the new feature made it into top importances
# # importances_v2 = pd.DataFrame({
# #     'feature': X_train_new.columns,
# #     'importance': model_v2.feature_importances_
# # }).sort_values('importance', ascending=False)

# # print("\nWhere does US_and_Senior rank?")
# # print(importances_v2[importances_v2['feature'] == 'US_and_Senior'])
# # print("\nTop 10 features now:")
# # print(importances_v2.head(10).to_string(index=False))


# # Train XGBoost with reasonable starting params
# # model_xgb = xgb.XGBRegressor(
# #     n_estimators=300,
# #     learning_rate=0.05,
# #     max_depth=6,
# #     subsample=0.8,
# #     colsample_bytree=0.8,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # model_xgb.fit(X_train_new, y_train)
# # y_pred_xgb = model_xgb.predict(X_test_new)

# # Evaluate
# # mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
# # rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# # r2_xgb = r2_score(y_test, y_pred_xgb)

# # residuals_xgb = y_test - y_pred_xgb
# # abs_residuals_xgb = np.abs(residuals_xgb)

# # error_df_xgb = pd.DataFrame({
# #     'actual': y_test.values,
# #     'predicted': y_pred_xgb,
# #     'error': residuals_xgb.values,
# #     'abs_error': abs_residuals_xgb.values
# # }).sort_values('abs_error', ascending=False)

# #
# model_v3 = xgb.XGBRegressor(
#     subsample=0.6,
#     reg_lambda=1.5,
#     reg_alpha=0,
#     n_estimators=500,
#     min_child_weight=3,
#     max_depth=10,
#     learning_rate=0.03,
#     colsample_bytree=0.6,
#     random_state=42,
#     n_jobs=-1
# )

# model_v3.fit(X_train_new, y_train)
# y_pred_v3 = model_v3.predict(X_test_new)

# mae_v3 = mean_absolute_error(y_test, y_pred_v3)
# rmse_v3 = np.sqrt(mean_squared_error(y_test, y_pred_v3))
# r2_v3 = r2_score(y_test, y_pred_v3)

# print("=== Previous best (tuned XGBoost) ===")
# print("MAE: 27120.50, RMSE: 40078.80, R²: 0.6235")

# print("\n=== With new features (skill_count + experience_bucket) ===")
# print(f"MAE: {mae_v3:.2f}")
# print(f"RMSE: {rmse_v3:.2f}")
# print(f"R² Score: {r2_v3:.4f}")

# # Check where new features rank in importance
# importances_v3 = pd.DataFrame({
#     'feature': X_train_new.columns,
#     'importance': model_v3.feature_importances_
# }).sort_values('importance', ascending=False)

# print("\nTop 15 features now:")
# print(importances_v3.head(15).to_string(index=False))










































# gemini train the model 

# import pandas as pd
# import numpy as np
# import re
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from collections import Counter
# from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
# from sklearn.linear_model import Ridge

# # 1. Load Data
# required_features = [
#     "YearsCodePro", "EdLevel", "DevType", "OrgSize", "Industry",
#     "RemoteWork", "Country", "LanguageHaveWorkedWith", "PlatformHaveWorkedWith",
#     "DatabaseHaveWorkedWith", "ToolsTechHaveWorkedWith", "ConvertedCompYearly"
# ]

# df = pd.read_csv("../data/survey_results_public.csv", usecols=required_features)

# df = df[df['ConvertedCompYearly'].notnull()].copy()
# df = df[df['LanguageHaveWorkedWith'].notnull()].copy()

# # --- TACTIC 1: Strict Domain Bounding ---
# # We limit the dataset to realistic global developer salaries to remove unexplainable noise
# df = df[(df['ConvertedCompYearly'] >= 10000) & (df['ConvertedCompYearly'] <= 250000)].copy()

# # 2. Skill Counts
# df['total_skills'] = (
#     df['LanguageHaveWorkedWith'].fillna('').apply(lambda x: len(x.split(';')) if x else 0) +
#     df['DatabaseHaveWorkedWith'].fillna('').apply(lambda x: len(x.split(';')) if x else 0) +
#     df['PlatformHaveWorkedWith'].fillna('').apply(lambda x: len(x.split(';')) if x else 0) +
#     df['ToolsTechHaveWorkedWith'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)
# )

# # 3. Clean Experience
# df['YearsCodePro'] = df['YearsCodePro'].replace({'Less than 1 year': 0, 'More than 50 years': 51})
# df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'], errors='coerce')
# df['YearsCodePro'] = df['YearsCodePro'].fillna(df['YearsCodePro'].median())

# # 4. Map Ordinals
# edlevel_mapping = {
#     "Primary/elementary school": 0, "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 1,
#     "Some college/university study without earning a degree": 2, "Associate degree (A.A., A.S., etc.)": 3,
#     "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 4, "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
#     "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 6, "Something else": 1
# }
# df['EdLevel_encoded'] = df['EdLevel'].map(edlevel_mapping).fillna(1)

# orgsize_mapping = {
#     "Just me - I am a freelancer, sole proprietor, etc.": 0, "2 to 9 employees": 1,
#     "10 to 19 employees": 2, "20 to 99 employees": 3, "100 to 499 employees": 4,
#     "500 to 999 employees": 5, "1,000 to 4,999 employees": 6, "5,000 to 9,999 employees": 7,
#     "10,000 or more employees": 8, "I don’t know": -1
# }
# df['OrgSize_encoded'] = df['OrgSize'].map(orgsize_mapping).fillna(-1)

# # 5. Extract Top Tech Flags
# def build_binary_flags(df, column, prefix, max_features=10):
#     all_items = df[column].dropna().str.split(';').explode()
#     top_items = pd.Series(Counter(all_items)).sort_values(ascending=False).head(max_features).index.tolist()
#     for item in top_items:
#         clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', item)
#         df[f"{prefix}_{clean_name}"] = df[column].apply(lambda x: 1 if pd.notnull(x) and item in x.split(';') else 0)
#     return df

# df = build_binary_flags(df, 'LanguageHaveWorkedWith', 'lang', max_features=15)
# df = build_binary_flags(df, 'DatabaseHaveWorkedWith', 'db', max_features=8)
# df = build_binary_flags(df, 'PlatformHaveWorkedWith', 'platform', max_features=5)

# # Keep Categoricals for Target Encoding
# columns_to_drop = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'EdLevel', 'OrgSize']
# df_final = df.drop(columns=columns_to_drop)

# # One-hot encode only RemoteWork
# df_final = pd.get_dummies(df_final, columns=['RemoteWork'], drop_first=True)

# # Clean Columns
# df_final.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df_final.columns]
# df_final = df_final[~df_final.duplicated()].reset_index(drop=True)

# # 6. Train Test Split
# X = df_final.drop(columns=['ConvertedCompYearly'])
# y = df_final['ConvertedCompYearly']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 7. Target Encoding
# target_cols = ['Country', 'DevType', 'Industry']
# for col in target_cols:
#     medians = y_train.groupby(X_train[col]).median()
#     global_median = y_train.median()
#     X_train[f'{col}_economic_base'] = X_train[col].map(medians).fillna(global_median)
#     X_test[f'{col}_economic_base'] = X_test[col].map(medians).fillna(global_median)
#     X_train = X_train.drop(columns=[col])
#     X_test = X_test.drop(columns=[col])

# # --- TACTIC 2: Diminishing Returns Feature ---
# X_train['Exp_Value_Multiplier'] = X_train['YearsCodePro'] * X_train['Country_economic_base']
# X_test['Exp_Value_Multiplier'] = X_test['YearsCodePro'] * X_test['Country_economic_base']

# # Polynomial feature for experience
# X_train['YearsCodePro_Squared'] = X_train['YearsCodePro'] ** 2
# X_test['YearsCodePro_Squared'] = X_test['YearsCodePro'] ** 2

# # --- TACTIC 3: Model Stacking ---
# # Base Model 1: Our tuned XGBoost
# xgb_model = xgb.XGBRegressor(
#     n_estimators=400,
#     learning_rate=0.04,
#     max_depth=6,
#     min_child_weight=10,      
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_alpha=2.0,           
#     reg_lambda=3.0,          
#     objective='reg:squarederror',
#     random_state=42,
#     n_jobs=-1
# )

# # Base Model 2: HistGradientBoosting (Excellent with categorical/dense numerical arrays)
# hgb_model = HistGradientBoostingRegressor(
#     max_iter=400,
#     learning_rate=0.04,
#     max_depth=6,
#     min_samples_leaf=15,
#     l2_regularization=2.0,
#     random_state=42
# )

# # Meta Model: Ridge Regression combines the predictions
# stacked_model = StackingRegressor(
#     estimators=[('xgb', xgb_model), ('hgb', hgb_model)],
#     final_estimator=Ridge(alpha=10.0),
#     n_jobs=-1
# )

# # Train the Stack
# stacked_model.fit(X_train, y_train)
# y_pred = stacked_model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print("\n=== Grandmaster Pipeline: Stacked Models + Strict Bounds ===")
# print(f"MAE:  ${mae:.2f}")
# print(f"RMSE: ${rmse:.2f}")
# print(f"R² Score: {r2:.4f}")


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








# claude train the model 

# import pandas as pd
# import numpy as np
# import re
# import xgboost as xgb
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from collections import Counter

# # ----------------------------------------------------------------------
# # 1. LOAD
# # ----------------------------------------------------------------------
# required_features = [
#     "YearsCodePro", "WorkExp", "EdLevel", "DevType", "OrgSize", "Industry",
#     "RemoteWork", "Country", "LanguageHaveWorkedWith", "PlatformHaveWorkedWith",
#     "DatabaseHaveWorkedWith", "ToolsTechHaveWorkedWith", "ConvertedCompYearly",
# ]

# df = pd.read_csv("../data/survey_results_public.csv", usecols=required_features)
# df = df[df['ConvertedCompYearly'].notnull()].copy()

# # Clip outliers (1%-99%)
# lower = df['ConvertedCompYearly'].quantile(0.01)
# upper = df['ConvertedCompYearly'].quantile(0.99)
# df = df[(df['ConvertedCompYearly'] >= lower) & (df['ConvertedCompYearly'] <= upper)].copy()

# # ----------------------------------------------------------------------
# # 2. KEY FIX: log-transform the target (salary is heavily right-skewed)
# # ----------------------------------------------------------------------
# df['LogComp'] = np.log1p(df['ConvertedCompYearly'])   # KEEP THIS - do not drop!

# # ----------------------------------------------------------------------
# # 3. YearsCodePro cleanup
# # ----------------------------------------------------------------------
# df['YearsCodePro'] = df['YearsCodePro'].replace({
#     'Less than 1 year': 0,
#     'More than 50 years': 51
# })
# df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'], errors='coerce')
# df['YearsCodePro'] = df['YearsCodePro'].fillna(df['YearsCodePro'].median())
# df = df.drop(columns=['WorkExp'])

# # NEW: experience buckets (real feature this time, not just a comment)
# df['ExpBucket'] = pd.cut(
#     df['YearsCodePro'],
#     bins=[-1, 2, 5, 10, 20, 100],
#     labels=['0-2', '3-5', '6-10', '11-20', '20+']
# )

# # ----------------------------------------------------------------------
# # 4. EdLevel mapping
# # ----------------------------------------------------------------------
# edlevel_mapping = {
#     "Primary/elementary school": "Primary",
#     "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary",
#     "Some college/university study without earning a degree": "Some College",
#     "Associate degree (A.A., A.S., etc.)": "Associate",
#     "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
#     "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
#     "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional/PhD",
#     "Something else": "Other"
# }
# df['EdLevel'] = df['EdLevel'].map(edlevel_mapping)
# df['EdLevel'] = df['EdLevel'].fillna('Unknown')          # impute instead of dropping rows

# # ----------------------------------------------------------------------
# # 5. RemoteWork / DevType — impute instead of dropping (keep sample size)
# # ----------------------------------------------------------------------
# df['RemoteWork'] = df['RemoteWork'].fillna('Unknown')
# df['DevType'] = df['DevType'].fillna('Unknown')

# # ----------------------------------------------------------------------
# # 6. OrgSize — FIX: separate "unknown" flag instead of polluting ordinal scale
# # ----------------------------------------------------------------------
# df = df[df['OrgSize'].notnull()].copy()
# orgsize_mapping = {
#     "Just me - I am a freelancer, sole proprietor, etc.": 0,
#     "2 to 9 employees": 1,
#     "10 to 19 employees": 2,
#     "20 to 99 employees": 3,
#     "100 to 499 employees": 4,
#     "500 to 999 employees": 5,
#     "1,000 to 4,999 employees": 6,
#     "5,000 to 9,999 employees": 7,
#     "10,000 or more employees": 8,
# }
# df['OrgSize_Unknown'] = (df['OrgSize'] == "I don’t know").astype(int)
# df['OrgSize_encoded'] = df['OrgSize'].map(orgsize_mapping)
# df['OrgSize_encoded'] = df['OrgSize_encoded'].fillna(df['OrgSize_encoded'].median())

# # ----------------------------------------------------------------------
# # 7. Country grouping
# # ----------------------------------------------------------------------
# top_countries = df['Country'].value_counts().head(15).index.tolist()
# df['Country_grouped'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')

# # ----------------------------------------------------------------------
# # 8. Industry
# # ----------------------------------------------------------------------
# df['Industry'] = df['Industry'].fillna('Not Specified')

# # ----------------------------------------------------------------------
# # 9. Language / Database / Platform / Tools -> binary flags + NEW skill_count
# # ----------------------------------------------------------------------
# df['LanguageHaveWorkedWith'] = df['LanguageHaveWorkedWith'].fillna('')
# df['DatabaseHaveWorkedWith'] = df['DatabaseHaveWorkedWith'].fillna('')
# df['PlatformHaveWorkedWith'] = df['PlatformHaveWorkedWith'].fillna('')
# df['ToolsTechHaveWorkedWith'] = df['ToolsTechHaveWorkedWith'].fillna('')

# all_languages = df['LanguageHaveWorkedWith'].str.split(';').explode()
# lang_counts = Counter([l for l in all_languages if l])
# all_lang_list = list(lang_counts.keys())
# for lang in all_lang_list:
#     col_name = f"lang_{lang.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '')}"
#     df[col_name] = df['LanguageHaveWorkedWith'].apply(lambda x: 1 if lang in x.split(';') else 0)

# all_dbs = df['DatabaseHaveWorkedWith'].str.split(';').explode()
# db_counts = Counter([d for d in all_dbs if d])
# top_databases = pd.Series(db_counts).sort_values(ascending=False).head(10).index.tolist()
# for db in top_databases:
#     col_name = f"db_{db.replace(' ', '_').replace('.', '')}"
#     df[col_name] = df['DatabaseHaveWorkedWith'].apply(lambda x: 1 if db in x.split(';') else 0)

# all_platforms = df['PlatformHaveWorkedWith'].str.split(';').explode()
# platform_counts = Counter([p for p in all_platforms if p])
# top_platforms = pd.Series(platform_counts).sort_values(ascending=False).head(8).index.tolist()
# for platform in top_platforms:
#     col_name = f"platform_{platform.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
#     df[col_name] = df['PlatformHaveWorkedWith'].apply(lambda x: 1 if platform in x.split(';') else 0)

# all_tools = df['ToolsTechHaveWorkedWith'].str.split(';').explode()
# tool_counts = Counter([t for t in all_tools if t])
# top_tools = pd.Series(tool_counts).sort_values(ascending=False).head(12).index.tolist()
# for tool in top_tools:
#     col_name = f"tool_{tool.replace(' ', '_').replace('(', '').replace(')', '').replace(chr(39), '')}"
#     df[col_name] = df['ToolsTechHaveWorkedWith'].apply(lambda x: 1 if tool in x.split(';') else 0)

# # NEW: skill_count — total number of distinct technologies known (strong salary signal)
# df['skill_count'] = (
#     df['LanguageHaveWorkedWith'].apply(lambda x: len([i for i in x.split(';') if i])) +
#     df['DatabaseHaveWorkedWith'].apply(lambda x: len([i for i in x.split(';') if i])) +
#     df['PlatformHaveWorkedWith'].apply(lambda x: len([i for i in x.split(';') if i])) +
#     df['ToolsTechHaveWorkedWith'].apply(lambda x: len([i for i in x.split(';') if i]))
# )

# # ----------------------------------------------------------------------
# # 10. Drop raw text columns, one-hot encode categoricals
# # ----------------------------------------------------------------------
# columns_to_drop = [
#     'Country', 'OrgSize', 'LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith',
#     'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'ConvertedCompYearly',
# ]
# df_final = df.drop(columns=columns_to_drop)

# categorical_cols = ['RemoteWork', 'EdLevel', 'DevType', 'Industry', 'Country_grouped', 'ExpBucket']
# df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)

# # ----------------------------------------------------------------------
# # 11. Split — target is now LogComp
# # ----------------------------------------------------------------------
# X = df_final.drop(columns=['LogComp'])
# y_log = df_final['LogComp']

# # Clean column names
# X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]

# # FIX: regex cleaning can cause two different original names to collide into
# # the same cleaned name (e.g. different special-char tokens both becoming "tool_C_").
# # This produces duplicate column labels, and XGBoost breaks with
# # "'DataFrame' object has no attribute 'dtype'" because data[col] returns a
# # DataFrame instead of a Series when the column name isn't unique.
# if X.columns.duplicated().any():
#     dup_names = X.columns[X.columns.duplicated()].unique().tolist()
#     print("Duplicate columns found after cleaning, merging:", dup_names)
#     # FAST fix: only touch the actual duplicate-named columns, not the whole
#     # DataFrame. Transposing the entire X (thousands of rows x hundreds of
#     # binary columns) is what caused the 3-5 minute hang.
#     for name in dup_names:
#         cols_idx = [i for i, c in enumerate(X.columns) if c == name]
#         merged = X.iloc[:, cols_idx].max(axis=1)   # OR-like merge for binary flags
#         X = X.drop(X.columns[cols_idx], axis=1)
#         X[name] = merged

# # FIX: XGBoost only accepts int/float/bool/category dtypes. get_dummies() and
# # the duplicate-column merge above can leave some columns as 'object' dtype,
# # which throws "DataFrame.dtypes for data must be int, float, bool or category".
# # Force every column to numeric as a safety net.
# for col in X.columns:
#     if X[col].dtype == 'object' or X[col].dtype == 'bool':
#         X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

# print("Any non-numeric columns left?", X.select_dtypes(include=['object']).columns.tolist())

# dup_mask = X.duplicated()
# X = X[~dup_mask].reset_index(drop=True)
# y_log = y_log[~dup_mask].reset_index(drop=True)

# X_train, X_test, y_train_log, y_test_log = train_test_split(
#     X, y_log, test_size=0.2, random_state=42
# )

# # Interaction feature
# senior_roles = [c for c in X.columns if 'Senior_Executive' in c or 'Engineering_manager' in c]
# us_col = [c for c in X.columns if 'United_States' in c]
# if senior_roles and us_col:
#     X_train['US_and_Senior'] = X_train[us_col[0]] * X_train[senior_roles].max(axis=1)
#     X_test['US_and_Senior'] = X_test[us_col[0]] * X_test[senior_roles].max(axis=1)

# # ----------------------------------------------------------------------
# # 12. Hyperparameter search (actually using RandomizedSearchCV now)
# # ----------------------------------------------------------------------
# param_dist = {
#     'n_estimators': [300, 500, 700],
#     'max_depth': [4, 6, 8],
#     'learning_rate': [0.01, 0.03, 0.05],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_lambda': [1, 1.5, 3],
#     'reg_alpha': [0, 0.5, 1],
#     'min_child_weight': [1, 3, 5],
# }

# base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# search = RandomizedSearchCV(
#     base_model, param_distributions=param_dist, n_iter=25,
#     scoring='r2', cv=3, random_state=42, n_jobs=-1, verbose=1
# )
# search.fit(X_train, y_train_log)
# best_model = search.best_estimator_
# print("Best params:", search.best_params_)

# # ----------------------------------------------------------------------
# # 13. Predict — remember to convert back from log scale for real R²/MAE
# # ----------------------------------------------------------------------
# y_pred_log = best_model.predict(X_test)

# y_pred_actual = np.expm1(y_pred_log)
# y_test_actual = np.expm1(y_test_log)

# mae = mean_absolute_error(y_test_actual, y_pred_actual)
# rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
# r2 = r2_score(y_test_actual, y_pred_actual)

# print("\n=== Final Results (original $ scale) ===")
# print(f"MAE: {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.4f}")

# # Also report R² directly on log scale (often the more honest metric for skewed targets)
# r2_log = r2_score(y_test_log, y_pred_log)
# print(f"R² Score (log scale): {r2_log:.4f}")

# importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': best_model.feature_importances_
# }).sort_values('importance', ascending=False)
# print("\nTop 15 features:")
# print(importances.head(15).to_string(index=False))

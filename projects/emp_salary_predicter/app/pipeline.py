import pandas as pd
import numpy as np
import re
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings('ignore')

class DeveloperSalaryPipeline:
    def __init__(self):
        # We will store all the "learned" rules here during training
        self.top_countries = []
        self.top_langs = []
        self.top_dbs = []
        self.top_platforms = []
        
        self.country_medians = {}
        self.industry_medians = {}
        self.profile_medians = {}
        self.global_median = 0
        
        self.model = None
        self.expected_columns = []
        
        # Static Mappings
        self.edlevel_mapping = {
            "Primary/elementary school": 0, "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 1,
            "Some college/university study without earning a degree": 2, "Associate degree (A.A., A.S., etc.)": 3,
            "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 4, "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
            "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 6, "Something else": 1
        }
        self.orgsize_mapping = {
            "Just me - I am a freelancer, sole proprietor, etc.": 0, "2 to 9 employees": 1,
            "10 to 19 employees": 2, "20 to 99 employees": 3, "100 to 499 employees": 4,
            "500 to 999 employees": 5, "1,000 to 4,999 employees": 6, "5,000 to 9,999 employees": 7,
            "10,000 or more employees": 8, "I don’t know": -1
        }

    def _preprocess_features(self, df, is_training=False):
        """Applies all data cleaning and feature engineering. Safe for both train and test data."""
        X = df.copy()
        
        # 1. Clean Experience
        X['YearsCodePro'] = X['YearsCodePro'].replace({'Less than 1 year': 0, 'More than 50 years': 51})
        X['YearsCodePro'] = pd.to_numeric(X['YearsCodePro'], errors='coerce').fillna(5) # Default 5 if missing
        
        # 2. Map Ordinals
        X['EdLevel_encoded'] = X['EdLevel'].map(self.edlevel_mapping).fillna(1)
        X['OrgSize_encoded'] = X['OrgSize'].map(self.orgsize_mapping).fillna(-1)
        
        # 3. Total Skills
        X['total_skills'] = (
            X.get('LanguageHaveWorkedWith', '').fillna('').apply(lambda x: len(str(x).split(';')) if x else 0) +
            X.get('DatabaseHaveWorkedWith', '').fillna('').apply(lambda x: len(str(x).split(';')) if x else 0) +
            X.get('PlatformHaveWorkedWith', '').fillna('').apply(lambda x: len(str(x).split(';')) if x else 0) +
            X.get('ToolsTechHaveWorkedWith', '').fillna('').apply(lambda x: len(str(x).split(';')) if x else 0)
        )
        
        # 4. Binary Flags (Learn during train, apply during test)
        def process_flags(data, column, prefix, max_feat, saved_list):
            if is_training:
                all_items = data[column].dropna().astype(str).str.split(';').explode()
                top_items = pd.Series(Counter(all_items)).sort_values(ascending=False).head(max_feat).index.tolist()
                saved_list.extend(top_items)
            
            for item in saved_list:
                clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', item)
                data[f"{prefix}_{clean_name}"] = data.get(column, '').fillna('').astype(str).apply(
                    lambda x: 1 if item in x.split(';') else 0
                )
            return data

        X = process_flags(X, 'LanguageHaveWorkedWith', 'lang', 12, self.top_langs)
        X = process_flags(X, 'DatabaseHaveWorkedWith', 'db', 6, self.top_dbs)
        X = process_flags(X, 'PlatformHaveWorkedWith', 'platform', 4, self.top_platforms)
        
        # 5. Grouped Country & Profiles
        if is_training:
            self.top_countries = X['Country'].value_counts().head(15).index.tolist()
            
        X['Country_grouped'] = X['Country'].apply(lambda x: x if x in self.top_countries else 'Other')
        X['Country_Role_Profile'] = X['Country_grouped'] + "_" + X['DevType'].fillna('Unknown')
        
        # 6. Drop Raw Columns
        cols_to_drop = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 
                        'ToolsTechHaveWorkedWith', 'EdLevel', 'OrgSize', 'Country', 'DevType']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        
        # 7. One-Hot Encode (RemoteWork)
        if 'RemoteWork' in X.columns:
            X = pd.get_dummies(X, columns=['RemoteWork'], drop_first=True)
            
        # Clean column names
        X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]
        
        return X

    def fit(self, df_train):
        print("Starting Pipeline Training...")
        
        # Split features and target
        y_train = df_train['ConvertedCompYearly']
        X_train = df_train.drop(columns=['ConvertedCompYearly'])
        
        # Process Features (This saves the top lists inside the class)
        X_train = self._preprocess_features(X_train, is_training=True)
        
        # Target Encoding
        self.global_median = y_train.median()
        
        self.country_medians = y_train.groupby(X_train['Country_grouped']).median().to_dict()
        self.industry_medians = y_train.groupby(X_train['Industry']).median().to_dict()
        self.profile_medians = y_train.groupby(X_train['Country_Role_Profile']).median().to_dict()
        
        X_train['Country_grouped_economic_base'] = X_train['Country_grouped'].map(self.country_medians).fillna(self.global_median)
        X_train['Industry_economic_base'] = X_train['Industry'].map(self.industry_medians).fillna(self.global_median)
        X_train['Country_Role_Profile_economic_base'] = X_train['Country_Role_Profile'].map(self.profile_medians).fillna(self.global_median)
        
        # Drop categorical strings used for encoding
        X_train = X_train.drop(columns=['Country_grouped', 'Industry', 'Country_Role_Profile'])
        
        # Advanced Features
        X_train['Exp_Value_Multiplier'] = X_train['YearsCodePro'] * X_train['Country_grouped_economic_base']
        X_train['YearsCodePro_Squared'] = X_train['YearsCodePro'] ** 2
        
        # Save exact column order for inference
        self.expected_columns = list(X_train.columns)
        
        # Train Models
        print("Training Stacking Ensemble (XGBoost + HistGradientBoosting)...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=350, learning_rate=0.05, max_depth=5, min_child_weight=8,      
            subsample=0.85, colsample_bytree=0.85, reg_alpha=3.0, reg_lambda=5.0,          
            objective='reg:squarederror', random_state=42, n_jobs=-1
        )

        hgb_model = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=3.0, random_state=42
        )

        self.model = StackingRegressor(
            estimators=[('xgb', xgb_model), ('hgb', hgb_model)],
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Training Complete!")
        return self

    def predict(self, df_test):
        # Process features using saved rules (is_training=False)
        X_test = self._preprocess_features(df_test, is_training=False)
        
        # Apply Saved Target Encodings
        X_test['Country_grouped_economic_base'] = X_test['Country_grouped'].map(self.country_medians).fillna(self.global_median)
        X_test['Industry_economic_base'] = X_test.get('Industry', pd.Series()).map(self.industry_medians).fillna(self.global_median)
        X_test['Country_Role_Profile_economic_base'] = X_test['Country_Role_Profile'].map(self.profile_medians).fillna(self.global_median)
        
        X_test = X_test.drop(columns=['Country_grouped', 'Industry', 'Country_Role_Profile'], errors='ignore')
        
        # Advanced Features
        X_test['Exp_Value_Multiplier'] = X_test['YearsCodePro'] * X_test['Country_grouped_economic_base']
        X_test['YearsCodePro_Squared'] = X_test['YearsCodePro'] ** 2
        
        # Ensure exact column match (fill missing with 0, drop unexpected)
        for col in self.expected_columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self.expected_columns]
        
        # Predict
        return self.model.predict(X_test)

    def save(self, filepath):
        joblib.dump(self, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
        return pipeline


# ==========================================
# USAGE EXAMPLE (Run this to test)
# ==========================================
if __name__ == "__main__":
    print("--- 1. LOADING DATA ---")
    required_features = [
        "YearsCodePro", "EdLevel", "DevType", "OrgSize", "Industry",
        "RemoteWork", "Country", "LanguageHaveWorkedWith", "PlatformHaveWorkedWith",
        "DatabaseHaveWorkedWith", "ToolsTechHaveWorkedWith", "ConvertedCompYearly"
    ]
    
    # Load and filter raw data as usual
    df = pd.read_csv("../data/survey_results_public.csv", usecols=required_features)
    
    # Strict Cleaning for Training Data
    df = df[df['ConvertedCompYearly'].notnull()].copy()
    df = df[df['LanguageHaveWorkedWith'].notnull()].copy()
    df = df[df['DevType'].notnull()].copy()
    df = df[df['Country'].notnull()].copy()
    
    # Temporarily group countries to apply Local IQR filtering safely before pipeline
    top_countries = df['Country'].value_counts().head(15).index.tolist()
    df['Temp_Country_Group'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')

    def filter_local_outliers(group):
        Q1 = group['ConvertedCompYearly'].quantile(0.15)
        Q3 = group['ConvertedCompYearly'].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.2 * IQR 
        upper_bound = Q3 + 1.2 * IQR
        return group[(group['ConvertedCompYearly'] >= lower_bound) & (group['ConvertedCompYearly'] <= upper_bound)]

    # Apply local economic bounding
    df = df.groupby('Temp_Country_Group', group_keys=False).apply(filter_local_outliers).reset_index(drop=True)
    df = df.drop(columns=['Temp_Country_Group'])
    
    # Train-test split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    print("\n--- 2. TRAINING & SAVING PIPELINE ---")
    # Instantiate our custom pipeline
    pipeline = DeveloperSalaryPipeline()
    
    # Fit the pipeline (This does all cleaning, encoding, and training automatically)
    pipeline.fit(df_train)
    
    # Save the pipeline to disk
    pipeline.save("developer_salary_model.pkl")
    
    print("\n--- 3. LOADING & PRODUCTION INFERENCE ---")
    # Imagine this is a totally different Python file (e.g., your FastAPI server app.py)
    loaded_pipeline = DeveloperSalaryPipeline.load("developer_salary_model.pkl")
    
    # Here is an example of raw data coming from a frontend Web Form (JSON)
    raw_user_input = [{
        "YearsCodePro": "5",
        "EdLevel": "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
        "DevType": "Developer, back-end",
        "OrgSize": "100 to 499 employees",
        "Industry": "Information Services, IT, Software Development, or other Technology",
        "RemoteWork": "Remote",
        "Country": "United States of America",
        "LanguageHaveWorkedWith": "Python;JavaScript;SQL",
        "PlatformHaveWorkedWith": "Amazon Web Services (AWS)",
        "DatabaseHaveWorkedWith": "PostgreSQL;Redis",
        "ToolsTechHaveWorkedWith": "Docker"
    }]
    
    # Convert incoming JSON to Pandas DataFrame
    incoming_data_df = pd.DataFrame(raw_user_input)
    
    # Predict directly! No manual string cleaning required.
    predicted_salary = loaded_pipeline.predict(incoming_data_df)
    
    print(f"\n✅ User Profile: 5 Years Exp, USA, Backend Developer (Python/SQL)")
    print(f"💰 Predicted Salary: ${predicted_salary[0]:,.2f}")
    
    # Verify overall performance on the test set
    y_test = df_test['ConvertedCompYearly']
    df_test_features = df_test.drop(columns=['ConvertedCompYearly'])
    
    y_pred = loaded_pipeline.predict(df_test_features)
    print("\n=== Pipeline Test Set Evaluation ===")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")


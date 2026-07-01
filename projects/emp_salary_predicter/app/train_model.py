import pandas as pd

df=pd.read_csv("../data/survey_results_public.csv")

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


# step 2 feature engineering 

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


# divide the feature based on there type 
def clean_experience(x):
    if pd.isna(x):
        return None
    x = str(x)
    if "Less than 1 year" in x:
        return 0.5
    elif "More than 50 years" in x:
        return 50
    else:
        try:
            return float(x)
        except:
            return None

def is_multivalue_column(series):
    sample = series.dropna().astype(str).head(50)
    
    # Count how many rows contain ';'
    count = sample.str.contains(';').sum()
    
    # If many rows have ';' → multi-value
    return count > 5

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
df['WorkExp'] = pd.to_numeric(df['WorkExp'], errors='coerce')
def devide_data_bytype(df,required_features, target_column):
    cat_features = []
    num_features = []
    multi_value_features = []

    for col in df[required_features]:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            num_features.append(col)

        elif pd.api.types.is_object_dtype(df[col]):
            if is_multivalue_column(df[col]):
                multi_value_features.append(col)
            else:
                cat_features.append(col)

    return {
        "numerical": num_features,
        "categorical": cat_features,
        "multi_value": multi_value_features,
        "target": target_column
    }
result=devide_data_bytype(df,required_features,"ConvertedCompYearly")
# print(devided_data_result)
# Numerical
df[result['numerical']] = df[result['numerical']].fillna(
    df[result['numerical']].median()
)

# Categorical
for col in result['categorical']:
    df[col] = df[col].fillna("Unknown")

# Multi-value
for col in result['multi_value']:
    df[col] = df[col].fillna("")


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
class MultiValueEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.mlbs = {}

    def fit(self, X, y=None):
        for col in self.columns:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col].fillna('').str.split(';'))
            self.mlbs[col] = mlb
        return self

    def transform(self, X):
        X = X.copy()
        all_encoded = []

        for col in self.columns:
            mlb = self.mlbs[col]
            split_data = X[col].fillna('').str.split(';')

            encoded = pd.DataFrame(
                mlb.transform(split_data),
                columns=[f"{col}_{c}" for c in mlb.classes_],
                index=X.index
            )

            all_encoded.append(encoded)

        # Drop original columns
        X = X.drop(columns=self.columns)

        # Combine all
        X = pd.concat([X] + all_encoded, axis=1)

        return X

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Column groups
num_cols = result['numerical']
cat_cols = result['categorical']
multi_cols = result['multi_value']

# Pipeline
pipeline = Pipeline(steps=[
    # Step 1: Multi-value encoding
    ("multi", MultiValueEncoder(columns=multi_cols)),
    # Step 2: ColumnTransformer
    ("preprocess", ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'  # keeps multi-value encoded columns
    ))
])

df = df[df['ConvertedCompYearly'].notna()].copy()
df = df.drop_duplicates()

df_model = df[required_features].copy()

X = df_model.drop('ConvertedCompYearly', axis=1)
y = df_model['ConvertedCompYearly']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# train the model  
linear_pipeline = Pipeline(steps=[
    
    ("multi", MultiValueEncoder(columns=multi_cols)),
    
    ("preprocess", ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )),
    
    ("model", LinearRegression())
])
# linear_pipeline.fit(X_train, y_train)
# y_pred = linear_pipeline.predict(X_test)
# print(y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print(min(y_pred), max(y_pred))

# print("Linear Regression Results:")
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

print(y.sort_values(ascending=False).head(10))

# plt.hist(y, bins=50)
# plt.title("Salary Distribution")
# # plt.show()
# plt.show()
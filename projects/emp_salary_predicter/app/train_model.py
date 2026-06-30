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

# Target
df = df[df['ConvertedCompYearly'].notna()]
df.drop_duplicates()

# print(required_features)
print(df[required_features])
print(df[required_features].isnull().sum())
# print(df[requred_features])
# print(required_features)
# print(len(required_features))

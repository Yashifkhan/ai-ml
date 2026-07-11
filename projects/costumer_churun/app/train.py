import pandas as pd 
df=pd.read_csv("../data/Customer-Churn.csv")

df=df.drop("customerID",axis=1)

# cat_coloumns=df.select_dtypes(include=["object"]).columns
# num_caolumns=df.select_dtypes(include=["int","float64"]).columns

# categorical column devide into int and float 

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

updition_col=[
     'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'TechSupport',
    "StreamingTV" ,      
    'StreamingMovies',
    'Churn'
]

for col in updition_col:
    df[col] = df[col].map({'Yes': 1, 'No': 0})


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod'], drop_first=True)

df["MultipleLines"] = df["MultipleLines"].fillna(0)
df["InternetService"] = df["InternetService"].fillna(0)
internet_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]
df[internet_cols] = df[internet_cols].fillna(0)
# cat_coloumns=df.select_dtypes(include=["object"]).columns
# num_caolumns=df.select_dtypes(include=["int","float64"]).columns


bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
corr = df.corr()["Churn"].sort_values()

df.drop_duplicates(inplace=True)
# print(corr)
# print(df["Churn"].value_counts(normalize=True))
# print(df.describe())
# print(df.duplicated().sum())
# corr = df.corr()["Churn"].sort_values()
# print(corr)

print(df["InternetService"].value_counts())
print(df["InternetService"].nunique())

# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# df.duplicated().sum()

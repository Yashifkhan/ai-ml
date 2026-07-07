import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# df=pd.read_csv("../data/SalaryData.csv")

df=pd.read_csv("../data/survey_results_public.csv")
print(df["RemoteWork"])

num_features=["experience_years","skills_count","salary","certifications"]
cat_features=["job_title","education_level","industry","company_size","location","remote_work"]


# sns.boxenplot(df["experience_years"])
# sns.kdeplot(df["experience_years"])
# plt.show()

# print(df["experience_years"].isnull().sum())
# print(df.describe())
# print(df.info())


# df["remote_work"] = df["remote_work"].astype("category")
# print(df.head())

# plt.scatter(df["experience_years"], df["salary"])
# plt.xlabel("Experience")
# plt.ylabel("Salary")
# plt.show()

# sns.boxplot(x="education_level", y="salary", data=df)
# plt.show()

# print(df.corr(numeric_only=True))
df = pd.get_dummies(df, columns=cat_features, drop_first=True)
X = df.drop("salary", axis=1)
y = df["salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))
# # print(df.isnull().sum())
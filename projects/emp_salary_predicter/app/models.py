import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("../data/SalaryData.csv")

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

plt.scatter(df["experience_years"], df["salary"])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# print(df.isnull().sum())
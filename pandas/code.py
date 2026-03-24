print("learn pandas")
import pandas as pd

# load the csv file 
# df=pd.read_csv("employee_data.csv")

# df=pd.read_json("product_data.json")
# load the json file 
# print("data id here",df)

# ferform the some operation on data 
df=pd.read_csv("raw_data.csv")

# remove the duplicate value in data
# df.drop_duplicates(inplace=True) 

# remove the null value in data 
# df.dropna()


# remove the column where any fild is null 
df.dropna(axis=1)

# calculate dteh avg of age column 
df.fillna(0)
df["age"] = df["age"].fillna(df["age"].mean())
print(df)
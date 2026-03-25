import pandas as pd

# df=pd.read_csv("raw_data.csv")    #load the data

# if emp salary is grater then 60k then apply 20% tax else 10 %
# df["tax"]=df["income"].apply([lambda x: "20%" if x>=60000 else "10"])

# add a new row 
# new_raw=pd.DataFrame([{"id":"11","name":"hello","age":"20","country":"india","icome":"20000"}])
# df=pd.concat([df,new_raw])  

# update the value any column our according 

# gender_map={"Male":"M","Female":"F","Unknown":"U"}
# df["gender"]=df["gender"].map(gender_map)


# give a increment to 1.5 % based on income 
# df=df.assign(new_income=df["income"] *1.5)

# replce the value of fild or column 
# df["country"]=df["country"].replace("USA","US")

# change the column index 

# df=df[["name","age","country","gender","income","id"]]

# df=df.copy()
# new_column=[col for col in df.columns if col != "id"] + ["id"]
# df=df[new_column]


# print(df.head())


df=pd.read_csv("raw_data.csv")

# df["age"].hist()
# print(df.head())

df.plot(kind="scatter",x="age",y="income")


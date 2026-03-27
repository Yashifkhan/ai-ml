import requests 
import pandas as pd

url="https://stephen-king-api.onrender.com/api/books"
resp=requests.get(url)
data=resp.json()
# print("resp is this ",data["data"])

df=pd.json_normalize(data["data"])
df=df[["id","Year","Title","Publisher","Notes","handle","ISBN"]]
print(df)

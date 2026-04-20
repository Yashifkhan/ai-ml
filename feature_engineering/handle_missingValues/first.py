# print("day one of handle the missing value in data")

# learn simple impuatater 
import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer

data = {
    "age": [25, 30, np.nan, 35, np.nan],
    "salary": [50000, 60000, 55000, np.nan, 65000]
}

df=pd.DataFrame(data)
# print("original data")
# print(df)

imputer=SimpleImputer(strategy="mean")
df_impute=imputer.fit_transform(df)

df_imputed=pd.DataFrame(df_impute,columns=df.columns)
print(df_imputed)
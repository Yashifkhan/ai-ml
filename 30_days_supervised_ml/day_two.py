#  Simple example of mean and mode and medain  
import pandas as pd

data = pd.DataFrame({
    "Age": [25, None, 30, 22],
    "Salary": [50000, 60000, None, 45000],
    "City": ["Delhi", "Mumbai", None, "Delhi"]
})

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Salary"] = data["Salary"].fillna(data["Salary"].mean())
data["City"] = data["City"].fillna(data["City"].mode()[0])

print(data)


# Handling Duplicate Data 
# Remove duplicates


import pandas as pd

data = pd.DataFrame({
    "Age": [25, 25, 30],
    "Salary": [50000, 50000, 60000],
    "City": ["Delhi", "Delhi", "Mumbai"]
})

data = data.drop_duplicates()
print(data)


# Encoding Categorical Data
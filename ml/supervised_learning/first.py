import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split



data=pd.read_csv("medical_data.csv")

# data show on graph 
sns.scatterplot(x=data["bmi"],y=data["charges"],hue=data["smoker"])

# plt.show()

X=data
y=data["charges"]

# data preprocess 
data=data.drop(columns=["charges","region"])
data["sex"]=data["sex"].map({"female":1,"male":0})
data["smoker"]=data["smoker"].map({"yes":1,"no":0})

# train test split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42) 
print(X_train.head())
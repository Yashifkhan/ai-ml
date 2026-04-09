import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from  sklearn.metrics import r2_score

# load the data 
data=pd.read_csv("medical_data.csv")

# data show on graph 
sns.scatterplot(x=data["bmi"],y=data["charges"],hue=data["smoker"])
# plt.show()

# data preprocess  encode
data["sex"] = data["sex"].map({"female":1, "male":0})
data["smoker"] = data["smoker"].map({"yes":1, "no":0})

# features
y=data["charges"]
X = data.drop(columns=["charges", "region"])

# train test split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train the modal 
modal=LinearRegression()
modal.fit(X_train,y_train)

# test and perdict the value 
y_perdict=modal.predict(X_test)

# check the score of modal 
# r2=r2_score(y_test,y_perdict)
# print("result of modal is after traning : ",r2)

print("result of modal ",y_perdict[0])
print("model is running wait for result")


# print(y_test.head())
# print(X_train.head())
# print(data.head())
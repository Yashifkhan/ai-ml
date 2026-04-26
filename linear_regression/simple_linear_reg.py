print("learn simple linear regression ")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# load the data 
df=pd.read_csv("std_placement.csv")

# plot the data on graph 
sns.scatterplot(
    x=df["Hours"],
    y=df["Scores"]
)
plt.title("Student Study hours scores ")


# extract the features 
X=df[["Hours"]]
y=df["Scores"]

# take row and column paramets 
# X=df.iloc[0:5]

# spllit the data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

# define the model & give the data to learn
model=LinearRegression()
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

sample = pd.DataFrame([[1]], columns=X_train.columns)
prediction = model.predict(sample)

print("predict score is : ",prediction)
plt.show()

# print(df.head())
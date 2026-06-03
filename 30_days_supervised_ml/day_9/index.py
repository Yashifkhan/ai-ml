# Project 1: Student Pass Prediction 
# pass or fail 

import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# load the data 
data = {
    "hours": [1,2,3,4,5,6,7,8],
    "pass":  [0,0,0,0,1,1,1,1]
}

df=pd.DataFrame(data)
# remove the first index 
# df=df.to_string(index=False)
# split the data 
X=df[["hours"]]
y=df["pass"]

model=LogisticRegression()
model.fit(X,y)


hours = [[5]]
y_predict_binary=model.predict(hours)
y_predict_probability=model.predict_proba(hours)


threshold = 0.5 
# prob = y_predict_binary[0]
prob = y_predict_probability[0][1]

if prob >= threshold:
    print("Pass the prob. of :" ,prob)
else:
    print("Fail the prob. of : ",prob)
    
# print("predict binary value:",y_predict_binary[0])
# print("predict the probability:",y_predict_probability[0])
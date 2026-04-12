import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data=pd.read_csv("salary_data.csv")

# input and output 
X=data.drop("salary" ,axis=1)
y=data["salary"]

# split data 
X_train,X_tast,y_train,y_tast=train_test_split( X,y,train_size=0.2)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_tast)

# user data with there skills
sample = pd.DataFrame( [[1,1,1,1,1,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0]],
    columns=["experience","js","react","node","python","ts","sql","mongodb","docker","aws","git","nextjs","tailwind","redux","graphql","kubernetes","cicd","testing","azure","agile"]
)
predicted_salary = model.predict(sample)
score=r2_score(y_tast,y_pred)
print("Accuracy:", round(score,2))
print("Predicted Salary:", round(predicted_salary[0],2))

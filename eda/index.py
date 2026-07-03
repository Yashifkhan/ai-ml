import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
df=pd.read_csv("train.csv")
# print(df.head())

# devide the data in category based on there type 
num_column="PassengerId","Age","Fare"
cat_column="Survived","Pclass","Sex","SibSp","Parch","Embarked"
mixed_coloumn="Name","Ticket","Cabin"

# discriptive statics ; work with single column 
# remove this column which is not use and not realtion any column 
#  work with numaric column 
df=df.drop("PassengerId",axis=1)

# print(df["Age"].describe())

# bar chart plot 
# sns.histplot(df["Age"])

# sns.boxplot(df["Age"])
# plt.show()

# if data have outlires then check it , is it really outlire or not check the data and then remove or use 
# print(df[df["Age"] > 65])

# conculation 
# data is normal distributed 
# data have outlirs 
# data have missing value 20%

# now move second num column is  ************
# Fare 
# check have the missing valaue or not 
# print(df["Fare"].isnull().sum())
# sns.boxplot(df[df["Fare"] > 250])
# plt.show()
# print(df[df["Fare"] > 250])

            #  univariant on categorical column                            
# ***************************step 2******************

print(df["Survived"].value_counts())
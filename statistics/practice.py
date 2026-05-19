import pandas as pd

data = pd.DataFrame({
    "area": [1200, 1500, 800, 1800, 2000, 950, 1700, 1600, 1100, 1400],
    "bedrooms": [2, 3, 2, 4, 4, 2, 3, 3, 2, 3],
    "age": [10, 5, 20, 2, 1, 15, 3, 7, 12, 6],
    "city": ["A", "A", "B", "A", "B", "B", "A", "A", "B", "B"],
    "price": [50, 65, 30, 80, 90, 35, 75, 70, 45, 60]
})
# Understand Data 

# step 1 
# get only first 5 row 
# print(data.head())

# get deep info like data type and more 
# print(data.info())

# give the stats of data like mean, std and mode
# print(data.describe())


# step 2 
# Central tendency (mean, median, mode)

# get the shap of data total row and column 
# print(data.shape)
# row,columns=data.shape
# print(row)
# print(column )

# mean 
# print(data['price'].mean())

# modain,midd value of data
# print(data['price'])
# print(data['price'].median())

# mode most frequent value in data (apply only on Categorical data)
# print(data['city'].mode()[0])

# step 3 Variance (Spread)
# print(data['price'].var())

# actual spread (std) 
# print(data['price'].std())
#    area  bedrooms  age city  price
# 0  1200         2   10    A     50
# 1  1500         3    5    A     65
# 2   800         2   20    B     30
# 3  1800         4    2    A     80
# 4  2000         4    1    B     90

# mera data mean 60 se approx 20 tak fala hai  
# # 19.72026594366539

# step 4 min max 
# print("max value",data['price'].min())
# print("min value",data['price'].max())

# step 5 range of data 
# range=max -min 
# max_val=data['price'].max() 
# min_val=data['price'].min()
# range_val=max_val-min_val
# print("range of your data:",range_val)


# step 6  Correlation (VERY IMPORTANT FOR ML)
# How strongly two variables are related
# 
# create the metrix table and get the max positive number 
# print(data.corr(numeric_only=True)) 

# correlation =data.corr(numeric_only=True)
# print(correlation)

# extract the price column 
# price_corr = correlation['price']

# remove the row of price 
# price_corr=price_corr.drop('price')
# print(price_corr)


# get max value of max correlated feature with target column 
# max_corr_feature_wit_target=price_corr.idxmax()
# max_corr_feature_value=price_corr.max()

# print("Feature:", max_corr_feature_wit_target)
# print("Correlation:", max_corr_feature_value)


# STEP 9: Group Analysis (Real Project Logic)
# print(data.groupby("city")["price"].mean())

# step 10  Distribution (Understanding Shape)

# import matplotlib.pyplot as plt


# plt.hist(data['price'])
# plt.plot(data['price'])
# plt.bar(data['price'])
# plt.show()

# import seaborn as sns 
# import matplotlib.pyplot as plt

# sns.heatmap(data.corr(numeric_only=True), annot=True)
# plt.show()

# sns.histplot(data['price'],kde=True)
# plt.show()

# sns.boxplot(x=data["price"])
# plt.show()

# sns.scatterplot(x=data["area"], y=data["price"])
# plt.show()


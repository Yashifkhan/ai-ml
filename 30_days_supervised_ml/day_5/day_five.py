

# guide 
# If $p \le 0.05$ (low), $H_0$ must go (Reject $H_0$ $\rightarrow$ Not Normal).
# If $p > 0.05$ (high), $H_0$ stays (Fail to Reject $H_0$ $\rightarrow$ Normal).


import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("house_price_regression_learning.csv")

# df["area_sqft"].fillna(df["area_sqft"].median(),inplace=True)
# df["bedrooms"].fillna(df["bedrooms"].median(),inplace=True)
# df["location_score"].fillna(df["location_score"].median(),inplace=True)
# df["house_age_years"].fillna(df["house_age_years"].median(),inplace=True)


# check the data is normal distrubuted or not 
# 1 step check the p value 
# p value > 0.05 then normal distrubuted  Safe for Z-score we apply z score
# p value <= 0.05 then not  normal distrubuted  It is skewed! Use IQR instead or first create distrubuted and then apply zscore


# this all part for remove the outlirs in data 

# *********** DATA IS NORMAL DISTRUBUTE THEN DIRECT APPLY Z-SCORE and check the data have outlirs or not  ********************
# CRITICAL: The Z-score formula math will crash or return 'NaN' if there are blank spaces!
# df_clean=df.dropna(subset=["location_score"]).copy()
# This calculates how many standard deviations away each row is from the average.
# raw_z_score=stats.zscore(df_clean["location_score"])
# abs_z_scores=np.abs(raw_z_score)
# z_threshold = 3
# outlier_condition = abs_z_scores > z_threshold
# outliers_detected = df_clean[outlier_condition]


# print("z-score result ")
# print("outliers in data ",outliers_detected)
# print("total outlires : ",len(outliers_detected))

# *********** DATA IS NORMAL DISTRUBUTE THEN DIRECT APPLY Z-SCORE ********************



        #************ DATA NOT NORMAL DISTRUBUTE AND YOU WANT TO APPLY Z SCORE THEN USE IT and real example of remove the ouliers in data and show on plot bot real data and remove outlirs then show on plot  *********
# if data is not normal distrubuted then first crete distribute then use stats.boxcox
# stats.boxcox  It returns two things: 
#   1. 'transformed_price': The new, normally distributed values.
#   2. 'lam': The optimal lambda power exponent it used for the math.
transformed_price, lam = stats.boxcox(df["price"])
_,p_transformed=stats.shapiro(transformed_price)

z_scores=stats.zscore(transformed_price)
threshold = 3
outliers = transformed_price[np.abs(z_scores) > threshold]
outlier_index = np.where(np.abs(z_scores) > 3)
original_outliers = df["price"].iloc[outlier_index]


clean_data  = transformed_price[abs(z_scores) <= threshold]
mask = np.abs(z_scores) <= threshold
clean_df = df[mask]

# draw plot  with real data and not remove the outlirs 
sns.scatterplot(x=df["area_sqft"],y=df["price"])
plt.show()

#draw plot with clean data and rempve the outlirs .
# sns.scatterplot(x=clean_df["area_sqft"],y=clean_df["price"])
# plt.show()
print("Outliers in original data:", original_outliers)
print("p_transpor value is ",p_transformed)
# after the apply stats.boxcox and check the p value is ok then apply z-score else not
       #************ DATA NOT NORMAL DISTRUBUTE THEN USE IT *********







        #***************** DATA NOT NORMAL DISTRUBUTE THEN YOU USE DIRECT IQR METHOD ***************
        
        
        #***************** DATA NOT NORMAL DISTRUBUTE THEN YOU USE DIRECT IQR METHOD ***************
        




# stat , p_value=stats.shapiro(df["price"])
# print("stat value : ",stat)
# print("p value : ",p_value)

# df=df.drop_duplicates()
# print(df.isnull().sum())
# print("duplicat total value is : ",df.duplicated().sum())
# print(df.head())
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
data = {
    "experience": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                   5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    
    "salary": [15000, 18000, 22000, 28000, 35000, 45000, 58000, 72000, 90000, 110000,
               135000, 165000, 200000, 240000, 285000, 335000, 390000, 450000, 520000, 600000]
}
df = pd.DataFrame(data)
sns.scatterplot(data=df, x="experience", y="salary")
X,y = df[["experience"]],df["salary"]
# -------- Linear Regression --------
model = LinearRegression()
model.fit(X, y)
y_linear = model.predict(X)
# -------- Polynomial Regression --------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model2 = LinearRegression()
model2.fit(X_poly, y)
y_poly_train = model2.predict(X_poly)
X_range = pd.DataFrame(
    np.linspace(X["experience"].min(), X["experience"].max(), 100),
    columns=["experience"]
)
X_range_poly = poly.transform(X_range)
y_poly = model2.predict(X_range_poly)
plt.scatter(X, y)
plt.plot(X, y_linear, label="Linear")
plt.plot(X_range, y_poly, label="Polynomial")
plt.legend()
plt.title("Linear vs Polynomial Regression")
exp=8
sample_data=pd.DataFrame([[exp]],
    columns=["experience"]
)
pred_value=model2.predict(poly.transform(sample_data))
accu=r2_score(y,y_poly_train)
print(f"Prediction (Polynomial) for {exp} exp : " , round(pred_value[0]))
print("Accurecy is :",round(accu))
plt.show()
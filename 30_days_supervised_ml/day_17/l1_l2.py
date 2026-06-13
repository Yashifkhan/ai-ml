import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# Features: [study_hours, sleep_hours, mobile_hours, gaming_hours]
X = np.array([
    [5, 7, 100, 50],
    [6, 6, 120, 60],
    [7, 8, 90, 40],
    [4, 5, 130, 70],
    [8, 7, 80, 30],
    [9, 6, 70, 20],
    [3, 5, 140, 80],
    [10, 8, 60, 10]
])

# Target (marks) → depends mostly on study + sleep
y = np.array([60, 65, 72, 55, 78, 85, 50, 90])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ridge l2
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)


# lasso l1 
lasso = Lasso(alpha=0.5)
lasso.fit(X_scaled, y)

print("Ridge Weights:", ridge.coef_)
print("Lasso Weights:", lasso.coef_)
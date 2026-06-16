
# Boosting is a technique where models are built sequentially.

# First model learns from data
# Second model learns from mistakes of first model
# Third model learns from mistakes of first + second
# This continues…

# Final prediction = sum of all weak models

# Instead of one strong model, we combine many weak learners (small decision trees).

# So accuracy keeps improving step by step.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=100,   # number of trees
    learning_rate=0.1,  # step size
    max_depth=3
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
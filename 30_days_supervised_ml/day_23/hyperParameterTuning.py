from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train knn model 
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3,5,7,9],
    'weights': ['uniform', 'distance']
}

# grid = GridSearchCV(knn, param_grid, cv=5)
# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))


# decision tree 
# dt = DecisionTreeClassifier()

# param_dist = {
#     'max_depth': [None, 5,10,15],
#     'min_samples_split': [2,5,10],
#     'criterion': ['gini', 'entropy']
# }

# random = RandomizedSearchCV(dt, param_dist, n_iter=5, cv=5)
# random.fit(X_train, y_train)

# print(random.best_params_)

# y_pred = random.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# Model 3: Random Forest
# rf = RandomForestClassifier()

# param_grid = {
#     'n_estimators': [50,100,200],
#     'max_depth': [None, 10,20],
#     'min_samples_split': [2,5]
# }

# grid = GridSearchCV(rf, param_grid, cv=5)
# grid.fit(X_train, y_train)


# print(grid.best_params_)

# y_pred = grid.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))


# Model 4: SVM
svm = SVC()

param_grid = {
    'C': [0.1,1,10],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(svm, param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred)) 
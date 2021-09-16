import sys
sys.path.append('D:/programs/ML/Algorithms')
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from ml_algorithms.supervised.linear import LinearRegression
from ml_algorithms.supervised.linear import GradientDescent
from ml_algorithms.supervised.neighbors import KNNRegressor
from ml_algorithms.supervised.tree import DecisionTreeRegressor
from ml_algorithms.supervised.ensemble import RandomForestRegressor
from ml_algorithms.supervised.boosting import GradientBoostingRegressor


X, y = make_regression(n_samples=1000, n_features=10, n_informative=10,
    n_targets=1, bias=0.5, noise=0.05)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def test_linear():
    model = LinearRegression(alpha=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Linear regression:", mean_squared_error(y_test, y_pred))

def test_gd():
    model = GradientDescent()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Gradient descent regression:", mean_squared_error(y_test, y_pred))

def test_knn():
    model = KNNRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("KNNRegressor:", mean_squared_error(y_test, y_pred))

def test_decision_tree():
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("DecisionTreeRegressor:", mean_squared_error(y_test, y_pred))

def test_random_forest():
    model = RandomForestRegressor(n_trees=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("RandomForestRegressor:", mean_squared_error(y_test, y_pred))

def test_gb():
    model = GradientBoostingRegressor(n_estimators=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("GradientBoostingRegressor:", mean_squared_error(y_test, y_pred))

def run_regression_test():
    test_linear()
    test_gd()
    test_knn()
    test_decision_tree()
    test_random_forest()
    test_gb()


if __name__ == "__main__":
    run_regression_test()
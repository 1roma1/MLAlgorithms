from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from ml_algorithms.supervised.linear.linear_regression import LinearRegression
from ml_algorithms.supervised.linear.gradien_descent import GradientDescent
from ml_algorithms.supervised.neighbors.knn import KNNRegressor

X, y = make_regression(n_samples=1000, n_features=10, n_informative=10,
    n_targets=1, bias=0.5, noise=0.05)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def test_linear_regression():
    model = LinearRegression(alpha=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Linear regression:", mean_squared_error(y_test, y_pred))

def test_gradient_descent():
    model = GradientDescent()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Gradient descent regression:", mean_squared_error(y_test, y_pred))

def test_knn_regressor():
    model = KNNRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("KNNRegressor:", mean_squared_error(y_test, y_pred))

def run_regression_test():
    test_linear_regression()
    test_gradient_descent()
    test_knn_regressor()


if __name__ == "__main__":
    run_regression_test()
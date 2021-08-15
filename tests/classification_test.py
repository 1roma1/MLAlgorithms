from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from ml_algorithms.supervised.linear.logistic_regression import LogisticRegression

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=8, n_classes=2, n_redundant=0
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def test_logistic_regression():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Logistic regression:", accuracy_score(y_test, y_pred))


def run_classification_test():
    test_logistic_regression()


if __name__ == "__main__":
    run_classification_test()
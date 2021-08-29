from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from ml_algorithms.supervised.linear.logistic_regression import LogisticRegression
from ml_algorithms.supervised.neighbors.knn import KNNClassifier
from ml_algorithms.supervised.bayes.naive_bayes import NaiveBayesClassifier
from ml_algorithms.supervised.svm.svm import SVM
from ml_algorithms.supervised.ensemble.decision_tree import DecisionTree
from ml_algorithms.supervised.ensemble.random_forest import RandomForest
from ml_algorithms.supervised.ensemble.adaboost import AdaBoost

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=8, n_classes=2, n_redundant=0
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def test_logistic_regression():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Logistic regression:", accuracy_score(y_test, y_pred))

def test_knn_classifier():
    model = KNNClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("KNNClassifier:", accuracy_score(y_test, y_pred))

def test_naive_bayes():
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("NaiveBayesClassifier:", accuracy_score(y_test, y_pred))

def test_svm():
    y_signed_train = (y_train * 2) - 1
    y_signed_test = (y_test * 2) - 1

    for kernel in ["linear", "rbf"]:
        model = SVM(max_iter=500, kernel=kernel)
        model.fit(X_train, y_signed_train)
        predictions = model.predict(X_test)
        print(f"SVMClassifier with {kernel} kernel:", accuracy_score(y_signed_test, predictions))

def test_decision_tree():
    model = DecisionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("DecisionTree:", accuracy_score(y_test, y_pred))

def test_random_forest():
    model = RandomForest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("RandomForest:", accuracy_score(y_test, y_pred))

def test_adaboost():
    y_signed_train = (y_train * 2) - 1
    y_signed_test = (y_test * 2) - 1

    model = AdaBoost(n_clf=50)
    model.fit(X_train, y_signed_train)
    y_pred = model.predict(X_test)
    print("AdaBoost:", accuracy_score(y_signed_test, y_pred))

def run_classification_test():
    test_logistic_regression()
    test_knn_classifier()
    test_naive_bayes()
    # test_svm()
    # test_decision_tree()
    # test_random_forest()
    # test_adaboost()


if __name__ == "__main__":
    run_classification_test()
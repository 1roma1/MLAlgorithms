import sys
sys.path.append('D:/programs/ML/Algorithms')
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from ml_algorithms.supervised.linear import LogisticRegression, SoftmaxRegression
from ml_algorithms.supervised.neighbors import KNNClassifier
from ml_algorithms.supervised.naive_bayes import NaiveBayesClassifier
from ml_algorithms.supervised.svm import LinearSVM
from ml_algorithms.supervised.tree import DecisionTreeClassifier
from ml_algorithms.supervised.ensemble import RandomForestClassifier
from ml_algorithms.supervised.boosting import AdaBoostClassifier


X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=8, n_classes=2, n_redundant=0
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def test_logistic_regression():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("LogisticRegression:", accuracy_score(y_test, y_pred))

def test_knn():
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
    model = LinearSVM()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("LinearSVM:", accuracy_score(y_test, predictions))

def test_decision_tree():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("DecisionTree:", accuracy_score(y_test, y_pred))

def test_random_forest():
    model = RandomForestClassifier(n_trees=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("RandomForest:", accuracy_score(y_test, y_pred))

def test_adaboost():
    y_signed_train = (y_train * 2) - 1
    y_signed_test = (y_test * 2) - 1

    model = AdaBoostClassifier(n_clf=50)
    model.fit(X_train, y_signed_train)
    y_pred = model.predict(X_test)
    print("AdaBoost:", accuracy_score(y_signed_test, y_pred))


def run_classification_test():
    test_logistic_regression()
    test_knn()
    test_naive_bayes()
    test_svm()
    test_decision_tree()
    test_random_forest()
    test_adaboost()


if __name__ == "__main__":
    run_classification_test()
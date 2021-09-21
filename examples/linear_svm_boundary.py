import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from ml_algorithms.supervised.svm import LinearSVM

iris = load_iris()

x = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.float64).reshape(-1, 1)

C = 2
model = LinearSVM(C=C, eta0=10, eta_d=1000, n_epochs=60000, random_state=2)
model.fit(x, y)

yr = y.ravel()

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.weights[0]
    b = svm_clf.bias[0]

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.plot(x[:, 0][yr==1], x[:, 1][yr==1], "g^", label="Iris virginica")
plt.plot(x[:, 0][yr==0], x[:, 1][yr==0], "bs", label="Not Iris virginica")
plot_svc_decision_boundary(model, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("Linear SVM", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
plt.savefig('examples/figures/linear_svm_boundary.png')
plt.show()
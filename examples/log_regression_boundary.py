import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from ml_algorithms.supervised.linear.logistic_regression import LogisticRegression

iris = load_iris()

x = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.int)

model = LogisticRegression()
model.fit(x, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2.7, 200).reshape(-1, 1)
)
X_new = np.c_[x0.ravel(), x1.ravel()]

plt.figure(figsize=(10, 4))
plt.plot(x[y==0, 0], x[y==0, 1], "bs")
plt.plot(x[y==1, 0], x[y==1, 1], "g^")
left_right = np.array([2.9, 7])

boundary = -(model.weights[0] * left_right + model.bias) / model.weights[1]

plt.plot(left_right, boundary, "r--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
plt.savefig('figures/log_regression_boundary.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons

from ml_algorithms.supervised.tree import DecisionTree

def plot_decision_boundary(clf, x, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    x_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(x_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(x[:, 0][y==0], x[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(x[:, 0][y==1], x[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(x[:, 0][y==2], x[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

deep_tree_clf1 = DecisionTree(min_samples_split=10)
deep_tree_clf1.fit(xm, ym)

plot_decision_boundary(deep_tree_clf1, xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title("Decision Tree Boundary", fontsize=14)
plt.savefig('examples/figures/decision_tree_boundary.png')
plt.show()
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

from ml_algorithms.supervised.linear import SoftmaxRegression

x, y = iris_data()
x = x[:, [0, 3]]

x[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

lr = SoftmaxRegression(lr=0.01, epochs=10, minibatches=1)
lr.fit(x, y)

plot_decision_regions(x, y, clf=lr)
plt.title('Softmax Regression Decision Boundaries')
plt.savefig('examples/figures/softmax_boundary.png')
plt.show()
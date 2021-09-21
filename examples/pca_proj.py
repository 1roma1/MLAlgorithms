import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from ml_algorithms.unsupervised.pca import PCA

iris = load_iris()
x = iris.data
y = iris.target

n_components = 2

pca = PCA(n_components=n_components)
pca.fit(x)
x_pca = pca.transform(x)

colors = ['navy', 'turquoise', 'darkorange']

for x_transformed, title in [(x_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(x_transformed[y == i, 0], x_transformed[y == i, 1],
                    color=color, lw=2, label=target_name)

    plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])
    
plt.savefig('examples/figures/pca_proj.png')
plt.show()
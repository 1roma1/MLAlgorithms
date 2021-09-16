import sys
sys.path.append('D:/programs/ML/Algorithms')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from ml_algorithms.unsupervised.kmeans import KMeans


data = load_iris()
x = data.data
y = data.target

y_pred = KMeans(n_clusters=3).predict(x)

plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.scatter(x[:, 2], x[:, 3], c="k", marker=".")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft=False)

plt.subplot(122)
plt.plot(x[y_pred==0, 2], x[y_pred==0, 3], "yo", label="Cluster 1")
plt.plot(x[y_pred==1, 2], x[y_pred==1, 3], "bs", label="Cluster 2")
plt.plot(x[y_pred==2, 2], x[y_pred==2, 3], "g^", label="Cluster 3")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)

plt.show()


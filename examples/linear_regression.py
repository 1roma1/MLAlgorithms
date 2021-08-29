import numpy as np
import matplotlib.pyplot as plt

from ml_algorithms.supervised.linear.linear_regression import LinearRegression

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.plot(x, y, 'bo')
plt.plot(x, y_pred, 'r-')
plt.savefig('figures/linear_regression_plot.png')
plt.show()
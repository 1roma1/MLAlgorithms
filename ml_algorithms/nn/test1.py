import numpy as np
import matplotlib.pyplot as plt

from train import *
from operation import *
from variable import Placeholder
from session import Session
from linear import Linear
from dense import Dense
from gradient_descent import GradientDescentOptimizer

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "bo")

linear = Linear()
linear.compile(optimizer=GradientDescentOptimizer, loss=squared_loss)

linear.fit(X, y, batch_size=10, epochs=10)

pred = linear.predict(X)

plt.plot(X, pred, "r-")
plt.show()
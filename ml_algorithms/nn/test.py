import matplotlib.pyplot as plt
import numpy as np
from variable import Variable, Placeholder
from operation import *
from train import *
from session import *
from sequential import Sequential
from dense import Dense
from gradient_descent import GradientDescentOptimizer

red_points = np.concatenate((
    0.2*np.random.randn(25, 2) + np.array([[0, 0]]*25),
    0.2*np.random.randn(25, 2) + np.array([[1, 1]]*25)
))


blue_points = np.concatenate((
    0.2*np.random.randn(25, 2) + np.array([[0, 1]]*25),
    0.2*np.random.randn(25, 2) + np.array([[1, 0]]*25)
))

plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
# plt.show()


X = Placeholder()


y = Placeholder()


model = Sequential([
    Dense(2, 2, activation=sigmoid),
    Dense(2, 2, activation=softmax)
])

model.compile(optimizer=GradientDescentOptimizer,
              loss=cross_entropy_loss)


feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    y:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

model.fit(feed_dict, X, y)

xs = np.linspace(-2, 2)
ys = np.linspace(-2, 2)
pred_classes = []
for x in xs:
    for y in ys:
        feed_dict={X: [[x, y]]}
        pred_class = model.predict(feed_dict, X)
        pred_classes.append((x, y, pred_class.argmax()))
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
# plt.show()
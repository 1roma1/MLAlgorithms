import random
import numpy as np

from variable import Placeholder
from session import Session
from variable import Variable
from operation import *
from train import compute_gradients

class Linear:
    def __init__(self):
        self.weight = Variable(np.random.randn(1, 1))
        self.bias = Variable(np.random.randn(1, 1))
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.session = None
            
    def forward(self, X):
        return add(matmul(X, self.weight), self.bias)
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.session = Session()

    def _data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))

        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
            yield features[batch_indices].reshape(batch_size, features.shape[1]), labels[batch_indices].reshape(batch_size, 1)

    def fit(self, X, y, batch_size, epochs):
        optimizer = self.optimizer()

        x_plh = Placeholder()
        y_plh = Placeholder()

        y_pred = self.forward(x_plh)
        J = self.loss(y_plh, y_pred)
                
        for epoch in range(epochs):
            for xi, yi in self._data_iter(batch_size, X, y):
                feed_dict = {x_plh: xi, y_plh: yi}
                J_value = self.session.run(J, feed_dict)
                
                gradients = compute_gradients(J)
                optimizer.apply_gradients(gradients)
            
            print("Epoch:", epoch, "Loss:", np.sum(J_value)/len(J_value))

    def predict(self, X):
        x_plh = Placeholder()
        feed_dict = {x_plh: X}
        y_pred = self.forward(x_plh)
        return self.session.run(y_pred, feed_dict)
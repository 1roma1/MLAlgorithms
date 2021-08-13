from operation import sigmoid
import numpy as np
from variable import Variable
from operation import matmul, add, sigmoid

class Dense:
    def __init__(self, in_units, units, activation):
        self.weight = Variable(np.random.randn(in_units, units))
        self.bias = Variable(np.random.randn(units,))
        self.activation = activation

    def forward(self, X):
        linear = add(matmul(X, self.weight), self.bias)
        return self.activation(linear)


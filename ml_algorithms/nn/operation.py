from graph import Graph
import numpy as np

class Operation:
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.consumers = []

        for input_node in input_nodes:
            input_node.consumers.append(self)

        Graph.get_instance().operations.append(self)

    def compute(self):
        pass

class Add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        self.inputs = [x_value, y_value]
        return x_value + y_value

class Sub(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        self.inputs = [x_value, y_value]
        return x_value - y_value

class Matmul(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        self.inputs = [a_value, b_value]
        return a_value.dot(b_value)

class Sigmoid(Operation):
    def __init__(self, a):
        super().__init__([a])

    def compute(self, a_value):
        return 1 / (1 + np.exp(-a_value))

class Softmax(Operation):
    def __init__(self, a):
        super().__init__([a])

    def compute(self, a_value):
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]

class CrossEntropyLoss(Operation):
    def __init__(self, y, y_pred):
        super().__init__([y, y_pred])

    def compute(self, y_value, y_pred_value):
        return neg(reduce_sum(reduce_sum(mul(y_value, log(y_pred_value)), axis=1)))

class Log(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x_value):
        return np.log(x_value)

class Multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value * y_value

class ReduceSum(Operation):
    def __init__(self, A, axis=None):
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        return np.sum(A_value, self.axis)

class Negative(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x_value):
        return -x_value

def add(x, y):
    return Add(x, y)

def sub(x, y):
    return Sub(x, y)

def mul(x, y):
    return Multiply(x, y)

def matmul(a, b):
    return Matmul(a, b)

def reduce_sum(a, axis=None):
    return ReduceSum(a, axis)

def neg(x):
    return Negative(x)
def sigmoid(a):
    return Sigmoid(a)

def softmax(a):
    return Softmax(a)

def log(x):
    return Log(x)

def cross_entropy_loss(y, y_pred):
    return neg(reduce_sum(reduce_sum(mul(y, log(y_pred)), axis=1)))

def squared_loss(y, y_pred):
    return mul(sub(y, y_pred), sub(y, y_pred))

from variable import Variable

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply_gradients(self, gradients):
        for node in gradients:
            if type(node) == Variable:
                grad = gradients[node]
                node.value -= self.learning_rate * grad
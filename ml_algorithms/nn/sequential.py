from variable import Placeholder
from session import Session
from operation import *
from train import compute_gradients

class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.session = None
            
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.session = Session()

    def fit(self, X, y, batch_size, epochs):
        optimizer = self.optimizer()

        m = X.shape[0]
                
        for epoch in range(epochs):

            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            x_plh = Placeholder()
            y_plh = Placeholder()

            y_pred = self.forward(x_plh)
            J = self.loss(y_plh, y_pred)
            
            for i in range(0, m, batch_size):
                xi = X[i:i+batch_size]
                yi = y[i:i+batch_size].reshape(batch_size, 1)

                feed_dict = {x_plh: xi, y_plh: yi}
                J_value = self.session.run(J, feed_dict)
                
                gradients = compute_gradients(J)
                optimizer.apply_gradients(gradients)
            
            print("Epoch:", epoch, "Loss:", J_value)

    def predict(self, feed_dict, X):
        y_pred = self.forward(X)
        return self.session.run(y_pred, feed_dict)[0]
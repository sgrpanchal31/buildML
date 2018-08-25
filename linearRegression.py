import numpy as np
import matplotlib.pyplot as plt

class linearClassifier(object):
    def __init__(self):
        self.W = None
    
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        
        num_train, dim = X.shape
        num_classes = 1 + np.max(y)

        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        
        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_ind = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W += -learning_rate*grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        
        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        y_pred = np.argmax(X.dot(self.W), axis=1)

        return y_pred

    def loss(self, X_batch, y_batch):
        


def step_gradient(b_current, k_current, train_x, train_y, learning_rate):
    b_grad = 0
    k_grad = 0
    n = len(train_y)
    for i in range(n):
        x = train_x[i]
        y = train_y[i]
        b_grad += (1/n) * (((k_current * x) + b_current) - y)
        k_grad += (1/n) * x * (((k_current * x) + b_current) - y)
    new_b = b_current - (learning_rate * b_grad)
    new_k = k_current - (learning_rate * k_grad)
    return [new_b, new_k]



def run():
    file = 'data/houses.csv'
    points = np.array(np.genfromtxt(file, delimiter=',', skip_header=1))
    # learning_rate = 0.0000001

    train_x = points[:,0]  # 
    train_y = points[:,1]  # prices
    init_b = 0
    init_k = 0             # y=k*x+b

    num_iters = 200
    [b, k] = run_descent(train_x, train_y, init_b, init_k, num_iters, learning_rate)
    print(b, k)
    drawPlot(train_x, train_y, k, b)

if __name__ == '__main__':
    run()
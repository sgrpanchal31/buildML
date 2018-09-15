import numpy as np

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.dot(X, W)

    

    
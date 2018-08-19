import numpy as np
import matplotlib.pyplot as plt


def drawPlot(train_x, train_y, m, b):
    plt.plot(train_x, train_y, 'ro')
    plt.plot([0, 7000], [0 + b, 7000*m + b], color='b', linestyle='-', linewidth=2)
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

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

def run_descent(train_x, train_y, init_b, init_k, num_iters, learning_rate):
    b = init_b
    k = init_k
    for i in range(num_iters):
        b, k = step_gradient(b, k, train_x, train_y, learning_rate)
    return [b, k]

def run():
    file = 'data/houses.csv'
    points = np.array(np.genfromtxt(file, delimiter=',', skip_header=1))
    learning_rate = 0.0000001

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
import matplotlib.pyplot as plt

def drawPlot(train_x, train_y, m, b):
    plt.plot(train_x, train_y, 'ro')
    plt.plot([0, 7000], [0 + b, 7000*m + b], color='b', linestyle='-', linewidth=2)
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()
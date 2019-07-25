import numpy as np

class Net:
    def __init__(self, layers):
        self.W = []
        self.b = []
        # -1 are appended to arrays in order to assign n-th index to n-th layer, e.g. W2 = W[2]
        self.W.append(-1)
        self.b.append(-1)
        for i in range(1,len(layers)):
            self.W.append(np.random.randn(layers[i], layers[i-1]) * 0.01)
            self.b.append(np.zeros((layers[i], 1)))

    def print_me(self):
        print(self.W[2])


if __name__ == '__main__':
    test1 = Net((5,4,1))
    test1.train()
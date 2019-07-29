import numpy as np

class Net:
    def __init__(self, layers):
        self.W = []
        self.b = []
        self.A_saved = []
        self.L = len(layers)
        # -1 are appended to arrays in order to assign n-th index to n-th layer, e.g. W2 = W[2]
        self.W.append(-1)
        self.b.append(-1)
        self.A_saved.append(-1)
        for i in range(1,self.L):
            self.W.append(np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1]))
            self.b.append(np.zeros((layers[i], 1)))
            self.A_saved.append(-1)
        self.W[-1] = self.W[-1] * 0.5   # different activation function in last layer

    def print_me(self):
        print(len(self.b[-1]))

    @staticmethod
    def activation(Z, type):
        if type == 'relu':
            return np.maximum(0,Z)
        elif type == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif type == 'softmax':
            temp = np.exp(Z)
            return temp/np.sum(temp)


    def train(self, X, Y):
        self.forward_prop(X)
        #cost_function()
        #backward_prop()
        #parameters_update()
        pass

if __name__ == '__main__':
    #np.random.seed(1)
    test1 = Net((5,4,1))
    X = np.array([0.05, 11, 100, 2, 0.8], ndmin=2)
    X = X.T
    #test1.forward_prop(X)
    test1.print_me()
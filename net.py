import numpy as np


class Net:
    def __init__(self, layers):
        """
            Create dictionaries for storing variables and initialize parameters
        """
        self.params = {}
        self.grads = {}
        self.saved = {}
        self.L = len(layers) - 1
        for i in range(1, self.L + 1):
            self.params['W' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            self.params['b' + str(i)] = np.zeros((layers[i], 1))

    def print_me(self):
        pass

    @staticmethod
    def activation(Z, act_type):
        if act_type == 'relu':
            return np.maximum(0,Z)
        elif act_type == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif act_type == 'softmax':
            temp = np.exp(Z)
            return temp/np.sum(temp)
        else:
            print('something went wrong')
            quit()

    def forward_prop(self, X):
        A = X
        for i in range(1, self.L):
            Z = np.dot(self.params['W' + str(i)], A) + self.params['b' + str(i)]
            A = Net.activation(Z, 'relu')
            self.saved['A' + str(i)] = A
            self.saved['Z' + str(i)] = Z
        # for last layer we use softmax or sigmoid accordingly to output layer size
        Z = np.dot(self.params['W' + str(self.L)], A) + self.params['b' + str(self.L)]
        if len(self.params['b' + str(self.L)] == 1):
            A = Net.activation(Z, 'sigmoid')
        else:
            A = Net.activation(Z, 'softmax')
        self.saved['A' + str(self.L)] = A
        self.saved['Z' + str(self.L)] = Z

    def cost_function(self, Y):
        m = Y.shape[1]
        cost1 = np.dot(Y, np.log(self.saved['A' + str(self.L)].T))
        cost2 = np.dot((1 - Y), np.log(1 - self.saved['A' + str(self.L)].T))
        cost_total = -1/m * (cost1 + cost2)
        cost_total = np.squeeze(cost_total)
        return cost_total

    def backward_prop(self, Y):
        m = Y.shape[1]
        dAL = - np.divide(Y, self.saved['A' + str(self.L)]) + np.divide((1 - Y), (1 - self.saved['A' + str(self.L)]))
        self.grads['dA' + str(self.L)] = dAL
        dZ = dAL * self.saved['A' + str(self.L)] * (1 - self.saved['A' + str(self.L)])
        dW = 1/m * np.dot(dZ, self.saved['A' + str(self.L - 1)].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.params['W' + str(self.L)].T, dZ)
        self.grads['dZ' + str(self.L)] = dZ
        self.grads['dW' + str(self.L)] = dW
        self.grads['db' + str(self.L)] = db
        self.grads['dA' + str(self.L)] = dA
        for i in reversed(range(1,self.L)):
            dZ = np.array(dA, copy=True)
            dZ[self.saved['Z' + str(i)] <= 0] = 0
            dW = 1/m * np.dot(dZ, self.saved['A' + str(i-1)].T)
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.params['W' + str(i)].T, dZ)
            self.grads['dZ' + str(i)] = dZ
            self.grads['dW' + str(i)] = dW
            self.grads['db' + str(i)] = db
            self.grads['dA' + str(i)] = dA

    def parameters_update(self, learning_rate):
        for i in range(1,self.L):
            self.params['W' + str(i)] = self.params['W' + str(i)] - learning_rate * self.grads['dW' + str(i)]
            self.params['b' + str(i)] = self.params['b' + str(i)] - learning_rate * self.grads['db' + str(i)]

    def train(self, X, Y, learning_rate, iter_no):
        self.saved['A0'] = X
        for i in range(iter_no):
            self.forward_prop(X)
            cost = self.cost_function(Y)
            self.backward_prop(Y)
            self.parameters_update(learning_rate)
            print(cost)



if __name__ == '__main__':
    #np.random.seed(1)
    test1 = Net((8, 5, 1))
    data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')
    Y = np.array(data[:,-1], ndmin=2)
    X = data[:,0:-1]
    X = X.T
    test1.train(X, Y, 0.1, 10000)

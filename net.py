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
        """
            Computes activation function g(Z)
                Uses only parameters Z and act_type (type of activation function)
                No changes made in instance variable
        """
        if act_type == 'relu':
            return np.maximum(0,Z)
        elif act_type == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif act_type == 'softmax':
            temp = np.exp(Z)
            return temp/np.sum(temp)
        else:
            print('something went wrong')
            exit(1)

    def forward_prop(self, X):
        """
            Performs forward propagation
                Uses parameter X (data) and instance variable 'params'
                Changes instance variable 'saved;
        """
        A = X
        for i in range(1, self.L):
            Z = np.dot(self.params['W' + str(i)], A) + self.params['b' + str(i)]
            A = Net.activation(Z, 'relu')
            self.saved['A' + str(i)] = A
            self.saved['Z' + str(i)] = Z
        # for last layer we use softmax or sigmoid accordingly to output layer size
        Z = np.dot(self.params['W' + str(self.L)], A) + self.params['b' + str(self.L)]
        act_type = 'sigmoid' if len(self.params['b' + str(self.L)] == 1) else 'softmax'
        A = Net.activation(Z, act_type)
        self.saved['A' + str(self.L)] = A
        self.saved['Z' + str(self.L)] = Z

    def cost_function(self, Y):
        """
            Computes cost function
                Uses parameter Y and instance variable 'saved' (predicted Y)
        """
        m = Y.shape[1]
        cost1 = np.dot(Y, np.log(self.saved['A' + str(self.L)].T))
        cost2 = np.dot((1 - Y), np.log(1 - self.saved['A' + str(self.L)].T))
        cost_total = -1/m * (cost1 + cost2)
        cost_total = np.squeeze(cost_total)
        return cost_total

    def backward_prop(self, Y):
        """
            Performs backward propagation
                Uses parameter Y and instance variables 'saved' and 'grads'
                Changes instance variable 'grads'
        """
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
        """
            Updates W and b
                Uses parameter 'learning_rate'
                Changes instance variable 'params'
        """
        for i in range(1,self.L):
            self.params['W' + str(i)] = self.params['W' + str(i)] - learning_rate * self.grads['dW' + str(i)]
            self.params['b' + str(i)] = self.params['b' + str(i)] - learning_rate * self.grads['db' + str(i)]

    def predict(self, features):
        """
            Gives prediction for given features
                Uses 'saved' instance variable
        """
        self.forward_prop(features)
        predictions = self.saved['A' + str(self.L)]
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        return predictions

    def train(self, X, Y, learning_rate, iter_no):
        """
            Trains the neural network using previously defined functions
                Uses and changes all instance variables
        """
        self.saved['A0'] = X
        for i in range(iter_no):
            self.forward_prop(X)
            #cost = self.cost_function(Y)
            self.backward_prop(Y)
            self.parameters_update(learning_rate)
            pred = self.predict(X)
            print(np.sum((pred == Y)/Y.shape[1])) if not i % 100 else 0


if __name__ == '__main__':
    test1 = Net((8, 10, 10, 1))
    data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')
    Y = np.array(data[:,-1], ndmin=2)
    X = data[:,0:-1]
    X = X.T
    test1.train(X, Y, 0.05, 50000)


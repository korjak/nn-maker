import numpy as np

class Net:
    def __init__(self, layers):
        self.W = []
        self.dW = []
        self.b = []
        self.db = []
        self.A_saved = []
        self.Z_saved = []
        self.dA = []
        self.dZ = []
        self.L = len(layers) - 1
        # -1 are appended to arrays in order to assign n-th index to n-th layer, e.g. W2 = W[2]
        self.W.append(-1)
        self.b.append(-1)
        self.A_saved.append(-1)
        self.Z_saved.append(-1)
        self.dA.append(-1)
        self.dW.append(-1)
        self.db.append(-1)
        self.dZ.append(-1)
        for i in range(1,self.L+1):
            #print(np.sqrt(1/layers[i-1]))
            self.W.append(np.random.randn(layers[i], layers[i-1]) * 0.01) #np.sqrt(2/layers[i-1]))
            self.b.append(np.zeros((layers[i], 1)))
            self.A_saved.append(-1)
            self.Z_saved.append(-1)
            self.dW.append(-1)
            self.db.append(-1)
            self.dA.append(-1)
            self.dZ.append(-1)
        #self.W[-1] = self.W[-1] * 0.5   # different activation function in last layer

    def print_me(self):
        print(self.dW)

    @staticmethod
    def activation(Z, type):
        if type == 'relu':
            return np.maximum(0,Z)
        elif type == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif type == 'softmax':
            temp = np.exp(Z)
            return temp/np.sum(temp)
        else:
            print('something went wrong')
            quit()

    def forward_prop(self, X):
        A = X
        for i in range(1, self.L):
            #print(i)
            Z = np.dot(self.W[i], A) + self.b[i]
            A = Net.activation(Z, 'relu')
            self.A_saved[i] = A
            self.Z_saved[i] = Z
        # for last layer we use softmax or sigmoid accordingly to output layer size
        Z = np.dot(self.W[-1], A) + self.b[-1]
        #print('jebanie ' + str(A))
        if len(self.b[-1]) == 1:
            A = Net.activation(Z, 'sigmoid')
        else:
            A = Net.activation(Z, 'softmax')
        #print(self.W[1].shape)
        self.A_saved[-1] = A
        #print('jebanie ' + str(A))
        #np.testing.assert_array_equal(self.A_saved[-1], self.A_saved[self.L])
        self.Z_saved[-1] = Z

    def cost_function(self, Y):
        m = Y.shape[1]
        np.testing.assert_array_equal(self.A_saved[-1], self.A_saved[self.L])
        cost1 = np.dot(Y, np.log(self.A_saved[-1]).T)
        #print(self.A_saved[-1].T)
        cost2 = np.dot((1 - Y), np.log(1 - self.A_saved[-1].T))
        cost_total = -1/m * (cost1 + cost2)
        cost_total = np.squeeze(cost_total)
        return cost_total

    def backward_prop(self, Y):
        m = Y.shape[1]
        #print('1 ' + str(1-self.A_saved[self.L]))
        dAL = - np.divide(Y, self.A_saved[self.L]) + np.divide((1 - Y), (1 - self.A_saved[self.L]))
        self.dA[self.L] = dAL
        dZ = dAL * self.A_saved[self.L] * (1 - self.A_saved[self.L])
        dW = 1/m * np.dot(dZ, self.A_saved[self.L-1].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.W[self.L].T, dZ)
        self.dZ[self.L] = dZ
        self.dW[self.L] = dW
        self.db[self.L] = db
        self.dA[self.L] = dA
        for i in reversed(range(1,self.L)):
            #print('kutangens: ' + str(i))
            dZ = np.array(dA, copy=True)
            #print('dZ.shape ' + str(dZ.shape))
            #print('Z.shape ' + str(self.Z_saved[i].shape))
            dZ[self.Z_saved[i] <= 0] = 0
            #print('i: ' + str(i))
            dW = 1/m * np.dot(dZ, self.A_saved[i-1].T)
            #print('dW.shape ' + str(dW.shape))
            #print('W.shape ' + str(self.W[i].shape))
            #print('A.shape ' + str(self.A_saved[i].shape))
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.W[i].T,dZ)
            self.dZ[i] = dZ
            self.dW[i] = dW
            self.db[i] = db
            self.dA[i] = dA

    def parameters_update(self, learning_rate):
        for i in range(1,self.L+1):
            #print(self.W[i].shape)
            #print(self.dW[i].shape)
            self.W[i] = self.W[i] - learning_rate * self.dW[i]
            self.b[i] = self.b[i] - learning_rate * self.db[i]

    def train(self, X, Y, learning_rate, iter_no):
        self.A_saved[0] = X
        for i in range(iter_no):
            self.forward_prop(X)
            cost = self.cost_function(Y)
            self.backward_prop(Y)
            self.parameters_update(learning_rate)
            print(cost)

if __name__ == '__main__':
    #np.seterr(divide='ignore', invalid='ignore')
    #np.random.seed(1)
    test1 = Net((8, 10, 10, 10, 1))
    #X = np.array([0.05, 11, 100, 2, 0.8], ndmin=2)
    #X = np.random.randn(10, 5)
    #Y = np.random.randint(2, size=(1, 10))
    #print(X)
    data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')
    Y = np.array(data[:,-1], ndmin=2)
    X = data[:,0:-1]
    X = X.T
    #print(X.shape)
    test1.train(X, Y, 0.08, 10000)
    #test1.forward_prop(X)
    #test1.cost_function(Y)
    #test1.backward_prop(Y)
    #test1.parameters_update(0.02)
    #test1.print_me()

import numpy as np
from math import floor


class Net:
    def __init__(self, layers):
        """
            Creates dictionaries for storing variables and initialize parameters
        """
        self.params = {}
        self.grads = {}
        self.saved = {}
        self.v = {}
        self.s = {}
        self.L = len(layers) - 1
        for i in range(1, self.L + 1):
            self.params['W' + str(i)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(1./layers[i-1])
            self.params['b' + str(i)] = np.zeros((layers[i], 1))
            self.v["dW" + str(i)] = np.zeros((self.params['W' + str(i)].shape[0], self.params['W' + str(i)].shape[1]))
            self.v["db" + str(i)] = np.zeros((self.params['b' + str(i)].shape[0], self.params['b' + str(i)].shape[1]))
            self.s["dW" + str(i)] = np.zeros((self.params['W' + str(i)].shape[0], self.params['W' + str(i)].shape[1]))
            self.s["db" + str(i)] = np.zeros((self.params['b' + str(i)].shape[0], self.params['b' + str(i)].shape[1]))

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
                Uses:
                     :param Y:  train set classes
                     instance variable 'saved'
                     instance variable 'grads'
                Changes:
                     instance variable 'grads'
        """
        m = Y.shape[1]
        dAL = - np.divide(Y, self.saved['A' + str(self.L)]) + np.divide((1 - Y), (1 - self.saved['A' + str(self.L)]))
        #print(dAL)
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
        for i in range(1, self.L):
            self.params['W' + str(i)] = self.params['W' + str(i)] - learning_rate * self.grads['dW' + str(i)]
            self.params['b' + str(i)] = self.params['b' + str(i)] - learning_rate * self.grads['db' + str(i)]

    def parameters_update_momentum(self, learning_rate, beta):
        """
            Updates W and b using momentum
                Uses:
                    :param learning_rate:   learning rate alpha specified by a user
                    :param beta:            beta coefficient specified by a user
                Changes:
                    instance variable 'v'
                    instance variable 'params'
        """
        for i in range(1, self.L):
            self.v['dW' + str(i)] = beta * self.v['dW' + str(i)] + (1 - beta) * self.grads['dW' + str(i)]
            self.v['db' + str(i)] = beta * self.v['db' + str(i)] + (1 - beta) * self.grads['db' + str(i)]
            self.params['W' + str(i)] = self.params['W' + str(i)] - learning_rate * self.v['dW' + str(i)]
            self.params['b' + str(i)] = self.params['b' + str(i)] - learning_rate * self.v['db' + str(i)]

    def parameters_update_adam(self, learning_rate, beta1, beta2, iteration):
        """
            Updates W and b using Adam optimizer
                Uses:
                    :param learning_rate:   learning rate alpha specified by a user
                    :param beta1:           beta coefficient for momentum, specified by a user
                    :param beta2:           beta coefficient for RMSprop, specified by a user
                    :param iteration:       number of current iteration
                Changes:
                    instance variable 'v'
                    instance variable 's'
                    instance variable 'params'
        """
        epsilon = 1e-8
        v_corrected = {}
        s_corrected = {}
        for i in range(1, self.L):
            self.v['dW' + str(i)] = beta1 * self.v['dW' + str(i)] + (1 - beta1) * self.grads['dW' + str(i)]
            self.v['db' + str(i)] = beta1 * self.v['db' + str(i)] + (1 - beta1) * self.grads['db' + str(i)]
            #print('tutaj tutaj tutaj tutaj tutaj tutaj ' + str(1 - beta1**iteration))
            #print(self.s['dW' + str(i)])
            v_corrected['dW'] = self.v['dW' + str(i)] / ((1 - beta1) ** i)
            v_corrected['db'] = self.v['db' + str(i)] / ((1 - beta1) ** i)
            self.s['dW' + str(i)] = beta2 * self.s['dW' + str(i)] + (1 - beta2) * self.grads['dW' + str(i)] ** 2
            self.s['db' + str(i)] = beta2 * self.s['db' + str(i)] + (1 - beta2) * self.grads['db' + str(i)] ** 2
            s_corrected['dW'] = self.s['dW' + str(i)] / ((1 - beta2) ** i)
            s_corrected['db'] = self.s['db' + str(i)] / ((1 - beta2) ** i)
            self.params['W' + str(i)] = self.params['W' + str(i)] - learning_rate * v_corrected['dW'] / (np.sqrt(s_corrected['dW']) + epsilon)
            self.params['b' + str(i)] = self.params['b' + str(i)] - learning_rate * v_corrected['db'] / (np.sqrt(s_corrected['db']) + epsilon)

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

    @staticmethod
    def make_batch_mini(X, Y, batch_size):
        examples_amount = X.shape[1]
        perm = np.random.permutation(examples_amount)
        X = X[:, perm]
        Y = Y[:, perm]
        batches_amount = floor(examples_amount/batch_size)
        batches = []
        prev_step = 0
        for i in range(batches_amount):
            next_step = prev_step + batch_size
            x_batch = X[:, prev_step:next_step]
            y_batch = Y[:, prev_step:next_step]
            batches.append((x_batch, y_batch))
            prev_step = next_step
        if examples_amount % batch_size != 0:
            already_batched = batch_size * batches_amount
            x_batch = X[:, already_batched:]
            y_batch = Y[:, already_batched:]
            batches.append((x_batch, y_batch))
        return batches

    def train(self, X, Y, learning_rate=0.05, beta1=0.9, beta2=0.999, batch_size=64, iter_no=10000):
        """
            Trains the neural network using previously defined functions
                Uses:
                    :param X:               feature vector
                    :param Y:               train set classes
                    :param learning_rate:   learning rate, specified by a user
                    :param beta1:           beta coefficient for momentum, specified by a user
                    :param beta2:           beta coefficient for RMSprop, specified by a user
                    :param batch_size:      size of one mini-batch
                    :param iter_no:         number of epochs
                Changes:
                    instance variables 'params'
                    instance variables 'grads'
                    instance variables 'saved'
                    instance variables 'v'
        """
        data = self.make_batch_mini(X, Y, batch_size)
        #print(data[1][1])
        Y_shuffled = [i for sublist in Y for i in sublist]
        #print(len(Y_shuffled))
        #predictions = []
        for i in range(iter_no):
            predictions = []
            for idx, batch in enumerate(data):
                self.saved['A0'] = batch[0]
                self.forward_prop(batch[0])
                #cost = self.cost_function(Y)
                self.backward_prop(batch[1])
                #self.parameters_update(learning_rate)
                #self.parameters_update_momentum(learning_rate, beta1)
                self.parameters_update_adam(learning_rate, beta1, beta2, i)
                pred = self.predict(batch[0])
                predictions.append(pred)
                #print(cost)
                #print('Epoch: ' + str(i) + '/' + str(iter_no))
                #print('Accuracy on current batch: ' + str(np.sum((pred == batch[1])/batch[1].shape[1])))
            print('Epoch: ' + str(i) + '/' + str(iter_no))
        predictions = [i.tolist() for i in predictions]
        predictions = [item for array in predictions for sublist in array for item in sublist]
        print('Accuracy: ' + str(np.sum(np.equal(predictions, Y_shuffled)/len(Y_shuffled))))



if __name__ == '__main__':
    #np.random.seed(555)
    test1 = Net((8, 10, 10, 1))
    data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')
    Y = np.array(data[:, -1], ndmin=2)
    X = data[:, 0:-1]
    X = X.T
    X = X - np.mean(X)
    X = X / np.var(X, ddof=1)
    test1.train(X, Y, 0.05, 0.9, 0.999, 64, 100)

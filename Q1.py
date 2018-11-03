import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split

# data = pd.read_csv('data.csv')
# label = pd.read_csv('labels.csv')



def cross_entropy_gradient(w, data, label):
    output = 1 / (np.ones((data.shape[0], 1)) + np.exp(-data.dot(w)))
    # return -(data.dot(label - 1 / (np.ones((data.shape[0], 1)) + np.exp(-data.dot(w)))))
    return np.dot(data.T, output - label) / data.shape[0]


#
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))
#
#
# def sigmoid_gradient(x):
#     return x.dot(np.ones(x.shape[0]) - x)

def GD(gradient, w, data, label):
    return w - eta * gradient(w, data, label)


def SGD(gradient, w, data, label):
    index = np.random.randint(0, data.shape[0] - 1, 1)
    return w - eta * gradient(w, data[index], label[index])


def SGDM(gradient, w, data, label, v):
    index = np.random.randint(0, data.shape[0] - 1, 1)
    return w - gamma * v - eta * gradient(w, data[index], label[index]), v


def adagrad(gradient, w, data, label, G):
    index = np.random.randint(0, data.shape[0] - 1, 1)
    gra = gradient(w, data[index], label[index])
    G += np.outer(gra, gra)
    delta = np.outer(eta / np.sqrt(np.diag(G) + np.ones(d) * epsilon), gra)

    return w - np.diag(delta).reshape(d, 1), G


def RMSprop(gradient, w, data, label, G):
    index = np.random.randint(0, data.shape[0] - 1, 1)
    gra = gradient(w, data[index], label[index])
    delta = np.outer(
        eta / np.sqrt(np.diag(gamma * np.mean(G) + (1 - gamma) * np.outer(gra, gra)) + np.ones(d) * epsilon), gra)
    G += np.outer(gra, gra)
    return w - np.diag(delta).reshape(d, 1), G


def adadelta(gradient, w_current, data, label, w_previous, delta_w, G):
    index = np.random.randint(0, data.shape[0] - 1, 1)
    gra = gradient(w_current, data[index], label[index])
    delta = np.outer(np.true_divide(np.sqrt(
        gamma * np.mean(delta_w) + (1 - gamma) * (w_current - w_previous) ** 2 + np.ones((data.shape[1], 1)) * epsilon),
                                    np.sqrt(np.diag(gamma * np.mean(G) + (1 - gamma) * np.outer(gra, gra)).reshape(
                                        data.shape[1], 1) + np.ones((data.shape[1], 1)) * epsilon)), gra)
    G += np.outer(gra, gra)
    delta_w += (w_current - w_previous) ** 2
    return w_current - np.diag(delta).reshape(data.shape[1], 1), w_current, delta_w, G


def adam(gradient, w, data, label, m, v, t):
    beta = 0.999
    index = np.random.randint(0, data.shape[0] - 1, 1)
    gra = gradient(w, data[index], label[index])
    m = gamma * m + (1 - gamma) * gra
    G = np.outer(gra, gra)
    v = beta * v + (1 - beta) * np.diag(G)
    m /= (1 - gamma ** t)
    v /= (1 - beta ** t)
    delta = np.outer((eta / (np.sqrt(v) + np.ones((data.shape[1], data.shape[1])) * epsilon)), m)

    return w - np.diag(delta).reshape(d, 1), m, v, t


class logistic:
    def __init__(self, x, y):
        self.data = x
        self.label = y
        # self.test_data = t
        # self.test_label = p
        self.error = epsilon
        self.weight = np.zeros(self.data.shape[1])
        self.output = np.zeros((self.data.shape[0], 1))
        self.process = []
        self.epoch = []

    def training_error(self, i, w):
        self.output = 1 / (
                np.ones((self.data.shape[0], 1)) + np.exp(-np.dot(self.data, w).reshape(self.data.shape[0], 1)))
        self.process.append(np.sum(
            -self.label * np.log(self.output + epsilon) - (np.ones((self.data.shape[0], 1)) - self.label) * np.log(
                np.ones((self.data.shape[0], 1)) - self.output + epsilon)) / self.data.shape[0])
        self.epoch.append(i)

    def GD_regression(self):
        w = np.ones((self.data.shape[1], 1))
        w_initial = np.zeros((self.data.shape[1], 1))
        i = 0
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            w = GD(cross_entropy_gradient, w, self.data, self.label)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def SGD_regression(self):
        w = np.ones((data.shape[1], 1))
        w_initial = np.zeros((self.data.shape[1], 1))
        i = 0
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            w = SGD(cross_entropy_gradient, w, self.data, self.label)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def SGDM_regression(self):
        w = np.ones((d, 1))
        v = 0
        w_initial = np.zeros((self.data.shape[1], 1))
        i = 0
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            [w, v] = SGDM(cross_entropy_gradient, w, self.data, self.label, v)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def Adagrad_regression(self):
        w = np.ones((d, 1))
        G = np.zeros((d, d))
        w_initial = np.zeros((self.data.shape[1], 1))
        i = 0
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            [w, G] = adagrad(cross_entropy_gradient, w, self.data, self.label, G)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def Adadelta_regression(self):
        w = np.ones((d, 1))
        delta_previous = np.zeros((d, 1))
        G = np.zeros((d, d))

        w_initial = np.zeros((self.data.shape[1], 1))
        i = 0
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            [w, w_initial, delta_previous, G] = adadelta(cross_entropy_gradient, w, self.data, self.label, w_initial,
                                                         delta_previous, G)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def RMSprop_regression(self):
        w = np.ones((d, 1))
        G = np.zeros((d, d))
        w_initial = np.zeros((self.data.shape[1], 1))
        i = 0
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            [w, G] = RMSprop(cross_entropy_gradient, w, self.data, self.label, G)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def Adam_regression(self):
        w = np.ones((d, 1))
        m = np.zeros((d, 1))
        v = np.zeros((d, 1))
        t = 1
        i = 0
        w_initial = np.zeros((self.data.shape[1], 1))
        while np.all(np.abs(w - w_initial) > epsilon):
            w_initial = w
            [w, m, v, t] = adam(cross_entropy_gradient, w, self.data, self.label, m, v, t)
            t += 1
            self.training_error(i, w)
            i += 1
        self.weight = w

if __name__ == '__main__':
    GDlist = []
    SGDlist = []
    SGDMlist = []
    Adagradlist = []
    RMSproplist = []
    Adadeltalist = []
    Adamlist = []
    N = 5000
    d = int(N * 0.1)
    # for i in range(5):
    data = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
    truth_weight = np.random.normal(0, 1, d)
    label = 1 / (np.ones((N, 1)) + np.exp(-np.dot(data, truth_weight)).reshape(N, 1)) + np.random.multivariate_normal(
        np.zeros(N), np.eye(N) * sigma, 1).reshape(N, 1)

    # train, test, label, test_label = train_test_split(data, label, test_size=0.33, random_state=42)
    # N = train.shape[0]
    # d = train.shape[1]

    classes = np.zeros((N, 1))
    classes[np.nonzero(label > 0.5)[0]] = 1
    classes[np.nonzero(label < 0.5)[0]] = 0
    label = classes

    label = 1 / (np.ones((N)) + np.exp(-np.dot(data, truth_weight))).reshape(N, 1) + np.random.multivariate_normal(
        np.zeros(N), np.eye(N) * sigma, 1).reshape(N, 1)

    processes = []
    epoches = []
    _logistic = logistic(data, label)
    _logistic.GD_regression()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)

    _logistic = logistic(data, label)
    _logistic.SGD_regression()
    # _logistic.test()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)

    _logistic = logistic(data, label)
    _logistic.SGDM_regression()
    # _logistic.test()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)
    _logistic = logistic(data, label)
    _logistic.Adagrad_regression()
    # _logistic.test()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)

    _logistic = logistic(data, label)
    _logistic.Adadelta_regression()
    # _logistic.test()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)

    _logistic = logistic(data, label)
    _logistic.RMSprop_regression()
    # _logistic.test()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)

    _logistic = logistic(data, label)
    _logistic.Adam_regression()
    # _logistic.test()
    processes.append(_logistic.process)
    epoches.append(_logistic.epoch)

    #plot
    epochs=max(len(e) for e in epoches)
    for i in processes:
        c=len(i)
    if c<epochs:
        for j in range(epochs-c):
            i.append(i[-1])
    epoch=[i for i in range(epochs)]
    for i in processes:
        plt.plot(epoch, i, label='GD')
        plt.plot(epoch, i, label='SGD')
        plt.plot(epoch, i, label='SGDM')
        plt.plot(epoch, i, label='Adagrad')
        plt.plot(epoch, i, label='Adadelta')
        plt.plot(epoch, i, label='RMSprop')
        plt.plot(epoch, i, label='Adam')
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')
    plt.legend(['GD','SGD','SGDM','Adagrad','Adadelta','RMSprop','Adam'])
    plt.show()

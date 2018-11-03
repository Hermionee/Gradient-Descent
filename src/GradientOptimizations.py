import numpy as np
import Utilities as util
import matplotlib.pyplot as plt
import pandas as pd
import time
epsilon=1e-8
sigma = 0.25
iterations = 300

class logistic:
    def __init__(self, x, y, grad, loss):
        self.data = x
        self.label = y
        # self.test_data = t
        # self.test_label = p
        self.error = epsilon
        self.weight = np.zeros(self.data.shape[1])
        self.output = np.zeros((self.data.shape[0], 1))
        self.process = []
        self.epoch = []
        self.GradientUtils = GradientOptimizations(loss, grad, 0.005, 0.9)
        # self.predictions = np.zeros(self.data.shape[0])

    def training_error(self, i, w):

        z=np.array(np.dot(self.data, w).reshape(self.data.shape[0], 1), dtype=np.float32)
        self.output = 1 / (
                np.ones((self.data.shape[0], 1)) + np.exp(-z))
        self.process.append(self.GradientUtils.loss(self.output, self.label))

        self.epoch.append(i)
#np.sum(-self.label * np.log(self.output + epsilon) - (np.ones((self.data.shape[0], 1)) - self.label) * np.log(np.ones((self.data.shape[0], 1)) - self.output + epsilon)) / self.data.shape[0]
        # self.predictions[np.nonzer(self.output > 0.5)[0]] = 1
        # self.predictions[self.output < 0.5] = 0

    def GD_regression(self):
        w = np.ones((self.data.shape[1], 1))
        i = 0
        while i < iterations:
            w = self.GradientUtils.GD(w, self.data, self.label)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def SGD_regression(self):
        w = np.ones((self.data.shape[1], 1))
        i = 0
        while i < iterations:
            np.random.shuffle(self.data)
            w = self.GradientUtils.SGD(w, self.data, self.label)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def SGDM_regression(self):
        w = np.ones((self.data.shape[1], 1))
        v = np.zeros((self.data.shape[1], 1))
        i = 0
        while i < iterations:
            np.random.shuffle(self.data)
            [w, v] = self.GradientUtils.SGDM(w, self.data, self.label, v)
            self.training_error(i, w)
            # print(self.GradientUtils.loss(self.output, self.label))
            i += 1
        self.weight = w

    def Adagrad_regression(self):
        w = np.ones((self.data.shape[1], 1))
        G = np.zeros((self.data.shape[1], self.data.shape[1]))
        i = 0
        while i<iterations:
            np.random.shuffle(self.data)
            [w, G] = self.GradientUtils.adagrad(w, self.data, self.label, G)
            self.training_error(i, w)
            i += 1
        self.weight = w

    def Adadelta_regression(self):
        w = np.ones((self.data.shape[1], 1))
        delta_previous = np.zeros((self.data.shape[1], 1))
        G = np.zeros((self.data.shape[1], self.data.shape[1]))
        i = 0
        w_initial = w
        while i < iterations:
            # if np.all(delta_previous) < 1e-2:
            #     delta_previous = np.zeros((self.data.shape[1], 1))
            #     w_initial = w
            np.random.shuffle(self.data)
            [w, w_initial, delta_previous, G, i] = self.GradientUtils.adadelta(w, self.data, self.label, w_initial, delta_previous, G, i)
            self.training_error(i, w)
        self.weight = w

    def RMSprop_regression(self):
        w = np.ones((self.data.shape[1], 1))
        G = np.zeros((self.data.shape[1], self.data.shape[1]))

        i = 0
        while i < iterations:
            np.random.shuffle(self.data)
            [w, G, i] = self.GradientUtils.RMSprop(w, self.data, self.label, G, i)
            self.training_error(i, w)
        self.weight = w

    def Adam_regression(self):
        w = np.ones((self.data.shape[1],1))
        m = np.zeros((self.data.shape[1], 1))
        v = np.zeros((self.data.shape[1], 1))
        t = 1
        i = 0
        while i < iterations:
            np.random.shuffle(self.data)
            [w, m, v] = self.GradientUtils.adam(w, self.data, self.label, m, v, t)
            self.training_error(i, w)
            t += 1
            i += 1

        self.weight = w


class GradientOptimizations(object):

    def __init__(self, loss, grad, eta, gamma):
        self.gradient = grad
        self.loss = loss
        self.eta = eta
        self.gamma = gamma

    def GD(self, w, data, label):
        return w - self.eta * self.gradient(w, data, label)

    def SGD(self,w, data, label):
        index = np.random.randint(0, data.shape[0] - 1, 1)
        return w - self.eta * self.gradient(w, data[index], label[index])

    def SGDM(self, w, data, label, v):
        index = np.random.randint(0, data.shape[0] - 1, 1)
        v = self.gamma * v + self.eta * self.gradient(w, data[index], label[index])
        return w - v, v

    def adagrad(self, w, data, label, G):
        index = np.random.randint(0, data.shape[0] - 1, 1)
        gra = self.gradient(w, data[index], label[index])
        G += np.outer(gra, gra)
        delta = np.outer(self.eta / np.sqrt(np.diag(G) + np.ones(data.shape[1]) * epsilon), gra)

        return w - np.diag(delta).reshape(data.shape[1], 1), G

    def RMSprop(self, w, data, label, G, iter):

        if iter != 0:
            G /= iter
        iter += 1
        index = np.random.randint(0, data.shape[0] - 1, 1)
        gra = self.gradient(w, data[index], label[index])
        delta = np.outer(
            self.eta / np.sqrt(np.diag(self.gamma * G + (1 - self.gamma) * np.outer(gra, gra)) + np.ones(data.shape[1]) * epsilon), gra)
        G += np.outer(gra, gra)
        return w - np.diag(delta).reshape(data.shape[1], 1), G, iter

    def adadelta(self, w_current, data, label, w_previous, delta_w, G, iter):

        if iter != 0:
            G /= iter
            delta_w /= iter
        iter += 1
        index = np.random.randint(0, data.shape[0] - 1, 1)
        gra = self.gradient(w_current, data[index], label[index])
        delta = np.outer(np.true_divide(np.sqrt(
            self.gamma * delta_w + (1 - self.gamma) * (w_current - w_previous) ** 2 + np.ones(
                (data.shape[1], 1)) * epsilon),
            np.sqrt(np.diag(self.gamma * G + (1 - self.gamma) * np.outer(gra, gra)).reshape(
                data.shape[1], 1) + np.ones((data.shape[1], 1)) * epsilon)), gra)
        G += np.outer(gra, gra)
        delta_w += (w_current - w_previous) ** 2
        return w_current - np.diag(delta).reshape(data.shape[1], 1), w_current, delta_w, G, iter

    def adam(self, w, data, label, m, v, t):
        beta = 0.1
        index = np.random.randint(0, data.shape[0] - 1, 1)
        gra = self.gradient(w, data[index], label[index])
        m = self.gamma * m + (1 - self.gamma) * gra
        G = np.outer(gra, gra)
        v = beta * v + (1 - beta) * np.diag(G).reshape((data.shape[1],1))

        m /= (1 - self.gamma ** t)
        v /= (1 - beta ** t)
        delta = np.outer((self.eta / (np.sqrt(v) + np.ones((data.shape[1],1)) * epsilon)), m)

        return w - np.diag(delta).reshape(data.shape[1], 1), m, v

if __name__ == '__main__':

    data = pd.read_csv('../data/iris.csv').values[:, 1:-1]
    label = pd.read_csv('../data/iris.csv').values[:, -1].reshape(data.shape[0],1)
    N = data.shape[0]
    d = data.shape[1]
    # for i in range(5):
    # N = 500
    # d = int(0.1*N)
    # data = np.random.multivariate_normal(np.arange(d) * 5, np.eye(d), N)
    # for row in data:
    #     indices = np.random.poisson(2)
    #     row[indices] = 0
    # truth_weight = np.random.normal(0, 1, d)
    # label = 1 / (np.ones((N, 1)) + np.exp(-np.dot(data, truth_weight)).reshape(N, 1)) + np.random.multivariate_normal(
    #     np.zeros(N), np.eye(N) * sigma, 1).reshape(N, 1)

    # train, test, label, test_label = train_test_split(data, label, test_size=0.33, random_state=42)
    # N = train.shape[0]
    # d = train.shape[1]
    #
    # classes = np.zeros((N, 1))
    # classes[np.nonzero(label > 0.5)[0]] = 1
    # classes[np.nonzero(label < 0.5)[0]] = 0
    # label = classes

    processes = []
    epoches = []
    time1 = time.time()
    _logisticGD = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticGD.GD_regression()
    processes.append(_logisticGD.process)
    epoches.append(_logisticGD.epoch)
    time2 = time.time()
    print("GD time: {0:.3f}s".format(time2-time1))

    time1=time.time()
    _logisticSGD = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticSGD.SGD_regression()
    # _logistic.test()
    processes.append(_logisticSGD.process)
    epoches.append(_logisticSGD .epoch)
    time2=time.time()
    print("SGD time: {0:.3f}s".format(time2-time1))

    time1=time.time()
    _logisticSGDM = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticSGDM.SGDM_regression()
    # _logistic.test()
    processes.append(_logisticSGDM.process)
    epoches.append(_logisticSGDM.epoch)
    time2=time.time()
    print("SGDM time: {0:.3f}s".format(time2-time1))

    time1=time.time()
    _logisticAdagrad = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticAdagrad.Adagrad_regression()
    # _logistic.test()
    processes.append(_logisticAdagrad.process)
    epoches.append(_logisticAdagrad.epoch)
    time2=time.time()
    print("Adagrad time: {0:.3f}s".format(time2-time1))

    time1=time.time()
    _logisticAdadelta = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticAdadelta.Adadelta_regression()
    # _logistic.test()
    processes.append(_logisticAdadelta.process)
    epoches.append(_logisticAdadelta.epoch)
    time2=time.time()

    print("Adadelta time: {0:.3f}s".format(time2-time1))

    time1=time.time()
    _logisticRMS = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticRMS.RMSprop_regression()
    # _logistic.test()
    processes.append(_logisticRMS.process)
    epoches.append(_logisticRMS.epoch)
    time2=time.time()
    print("RMSprop time: {0:.3f}s".format(time2-time1))

    time1=time.time()
    _logisticAdam = logistic(data, label, util.Metrics.ce_grad, util.Metrics.ce)
    _logisticAdam.Adam_regression()
    # _logistic.test()
    processes.append(_logisticAdam.process)
    epoches.append(_logisticAdam.epoch)
    time2=time.time()
    print(_logisticAdam.process)
    print("Adam time: {0:.3f}s".format(time2-time1))

    #plot
    epochs=max(len(e) for e in epoches)
    for i in processes:
        c = len(i)
        if c < epochs:
            for j in range(epochs-c):
                i.append(i[-1])
    epoch = [i for i in range(epochs)]
    for i in processes:
        plt.plot(epoch[1:], i[1:])
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend(['GD', 'SGD','SGDM','Adagrad','Adadelta','RMSprop','Adam'])
    plt.show()

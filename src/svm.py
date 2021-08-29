import math
from enum import Enum

import numpy as np

from subgradient_descent import SubgradientDescent


class LossFunction(Enum):
    HINGE = 0
    LOGISTIC = 1
    QUADRATIC = 2


class SupportVectorMachine:

    def __init__(self, loss):
        self.loss = loss
        self.regularizer = 10e-3
        self.parameters = None

    def __gradient(self, batch_size):
        if self.loss == LossFunction.HINGE:
            indicator = lambda x, y, w: x.dot(w) * y < 1
            return lambda x, y, w: self.regularizer * w - \
                                   sum(x[i] * y[i] * (1 / batch_size)
                                       if indicator(x[i], y[i], w) else 0 for i in range(batch_size))
        if self.loss == LossFunction.LOGISTIC:
            return lambda x, y, w: self.regularizer * w - \
                                   sum(x[i] * (y[i] / (1 + math.exp(-y[i] * x[i].dot(w)))) * (1 / batch_size)
                                       if -y[i] * x[i].dot(w) < 500 else 0 for i in range(batch_size))
        if self.loss == LossFunction.QUADRATIC:
            return lambda x, y, w: self.regularizer * w - sum(2 * (1 - x[i].dot(w) * y[i]) * (1 / batch_size) *
                                                              x[i] * y[i] for i in range(batch_size))

        raise ValueError("invalid loss function")

    def fit(self, train_x, train_y, batch_size, iterations):
        descent = SubgradientDescent(self.__gradient(batch_size), lambda i: 1 / (i + 1), iterations)
        opt = descent.execute(train_x, train_y, iterations)
        self.parameters = opt
        print(opt)
        return opt

    def evaluate(self, test_x, test_y):
        total = len(test_x)
        right = 0

        for i in range(len(test_x)):
            if test_x[i].dot(self.parameters) * test_y[i] > 0:
                right += 1

        return right / total

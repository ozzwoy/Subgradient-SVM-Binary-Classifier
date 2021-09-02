import math
from enum import Enum
from functools import partial

import numpy as np

from subgradient_descent import SubgradientDescent


class LossFunction(Enum):
    HINGE = 0
    LOGISTIC = 1
    QUADRATIC = 2

    @staticmethod
    def by_name(name):
        if name == "hinge":
            return LossFunction.HINGE
        elif name == "logistic":
            return LossFunction.LOGISTIC
        elif name == "quadratic":
            return LossFunction.QUADRATIC
        else:
            raise ValueError("invalid loss function")

    def value_at(self, x, y, w, _lambda):
        size = len(y)
        regular_term = w.dot(w) * _lambda
        mean = 0

        if self == LossFunction.HINGE:
            for i in range(size):
                margin = x[i].dot(w) * y[i]
                if margin < 1:
                    mean += (1 - margin) / size
        elif self == LossFunction.LOGISTIC:
            for i in range(size):
                margin = x[i].dot(w) * y[i]
                mean += math.log(1 + math.exp(-margin)) / size
        elif self == LossFunction.QUADRATIC:
            for i in range(size):
                margin = x[i].dot(w) * y[i]
                mean += (1 - margin) ** 2 / size
        else:
            raise ValueError("invalid loss function")

        return regular_term + mean

    def subgradient_at(self, x, y, w, _lambda):
        size = len(y)
        regular_term = 2 * _lambda * w
        mean = 0

        if self == LossFunction.HINGE:
            for i in range(size):
                margin = x[i].dot(w) * y[i]
                if margin < 1:
                    mean += -x[i] * y[i] / size
        elif self == LossFunction.LOGISTIC:
            for i in range(size):
                margin = x[i].dot(w) * y[i]
                mean += -(y[i] / (1 + math.exp(margin))) * x[i] / size
        elif self == LossFunction.QUADRATIC:
            for i in range(size):
                margin = x[i].dot(w) * y[i]
                mean += -2 * (1 - margin) * y[i] * x[i] / size
        else:
            raise ValueError("invalid loss function")

        return regular_term + mean


class SupportVectorMachine:

    def __init__(self, loss, iterations, batch_size=None):
        self.loss = loss
        self.iterations = iterations
        self.batch_size = batch_size
        self.regularizer = 1e-4
        self.optimizer = None
        self.history = []

    def fit(self, train_x, train_y):
        if self.loss == LossFunction.HINGE:
            step_size_rule = lambda iteration, x, y, w, calc_function, calc_gradient: 1 / (iteration + 1)
        else:
            step_size_rule = backtracking_line_search

        if self.batch_size is None:
            descent = SubgradientDescent(partial(self.loss.value_at, _lambda=self.regularizer),
                                         partial(self.loss.subgradient_at, _lambda=self.regularizer),
                                         step_size_rule)
        else:
            descent = SubgradientDescent(partial(self.loss.value_at, _lambda=self.regularizer),
                                         partial(self.loss.subgradient_at, _lambda=self.regularizer),
                                         step_size_rule, self.batch_size)

        opt = descent.execute(train_x, train_y, self.iterations)
        self.optimizer = opt
        self.history = descent.get_last_search_history()

    def evaluate(self, test_x, test_y):
        total = len(test_y)
        right = 0

        for i in range(total):
            margin = test_x[i].dot(self.optimizer) * test_y[i]
            if margin > 0:
                right += 1

        return right / total

    def get_optimizer(self):
        return self.optimizer

    def get_optimization_history(self):
        return self.history


def backtracking_line_search(iteration, x, y, w, calc_function, calc_gradient):
    alpha = 0.3
    beta = 0.8
    t = 1
    f_x = calc_function(x, y, w)
    d_x = calc_gradient(x, y, w)
    d_x_squared = d_x.dot(d_x)

    while calc_function(x, y, w - t * d_x) > f_x - t * alpha * d_x_squared:
        t = beta * t

    return t

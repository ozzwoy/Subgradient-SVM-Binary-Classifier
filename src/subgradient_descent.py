import random

import numpy as np


class SubgradientDescent:

    def __init__(self, calc_function, calc_subgradient, calc_step_size, batch_size=None):
        self.calc_function = calc_function
        self.calc_subgradient = calc_subgradient
        self.calc_step_size = calc_step_size
        self.batch_size = batch_size
        self.history = []

    def execute(self, x, y, iterations):
        joined = np.append(x, np.array([y]).transpose(), axis=1).tolist()
        current = np.zeros(len(x[0]))
        self.history.append(current)

        for i in range(iterations):
            if self.batch_size is not None:
                batch = self.__get_batch(joined)
                subgradient = self.calc_subgradient(batch[:, :-1], np.array(batch[:, -1]).flatten(), current)
            else:
                subgradient = self.calc_subgradient(x, y, current)

            step = self.calc_step_size(i, x, y, current, self.calc_function, self.calc_subgradient)
            current = current - step * subgradient
            self.history.append(current)

        return current

    def __get_batch(self, data):
        return np.array(random.sample(data, self.batch_size))

    def get_last_search_history(self):
        return self.history

import numpy as np


class SubgradientDescent:

    def __init__(self, get_subgradient, get_step_size, batch_size):
        self.get_subgradient = get_subgradient
        self.get_step_size = get_step_size
        self.batch_size = batch_size

    def execute(self, x, y, iterations):
        current = np.zeros(len(x[0]))

        for i in range(iterations):
            batch = self.__get_batch(x, y)
            subgradient = self.get_subgradient(x, y, current)
            current = current - self.get_step_size(i) * subgradient
            norm = np.linalg.norm(current)
            if norm != 0:
                current /= norm

        return current

    def __get_batch(self, x, y):
        joined = np.append(x, np.array([y]).transpose(), axis=1)
        if self.batch_size > 1:
            np.random.shuffle(joined)
            return joined[:self.batch_size, :]
        else:
            return [joined[np.random.randint(0, self.batch_size)]]

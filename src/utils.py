import math
import time

import numpy as np


def train_test_split(data_array, test_part):
    x = np.delete(data_array, -1, 1)
    y = data_array[:, -1]

    len_test = math.floor(len(x) * test_part)
    len_train = len(x) - len_test
    x_train, x_test = np.split(x, [len_train])
    y_train, y_test = np.split(y, [len_train])

    return x_train, y_train, x_test, y_test


def cross_validation(model, data_array, folds):
    np.random.shuffle(data_array)
    split = np.array_split(data_array, folds)
    mean_accuracy = 0
    mean_time = 0

    for iteration in range(folds):
        test_array = split[iteration]
        split_arrays = np.delete(split, iteration)
        train_array = np.concatenate(split_arrays)
        x_train, y_train = np.delete(train_array, -1, 1), train_array[:, -1]
        x_test, y_test = np.delete(test_array, -1, 1), test_array[:, -1]

        start = time.time()
        model.fit(x_train, y_train)
        end = time.time()
        elapsed = end - start

        accuracy = model.evaluate(x_test, y_test)
        mean_accuracy += accuracy / folds
        mean_time += elapsed / folds

    return mean_accuracy, mean_time
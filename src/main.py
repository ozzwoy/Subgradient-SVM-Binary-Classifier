import math
import time

import matplotlib.pyplot as plt
import numpy as np


from data_extractor import prepare_data
from svm import SupportVectorMachine, LossFunction


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


def create_measures_plot(title, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()


def test_loss(name, iterations, folds, data_array, test_part):
    train_x, train_y, test_x, test_y = train_test_split(data_array, test_part)

    loss = LossFunction.by_name(name)
    model = SupportVectorMachine(loss, iterations[-1])
    model.fit(train_x, train_y)
    history = model.get_optimization_history()
    values = [model.loss.value_at(train_x, train_y, history[j], model.regularizer) for j in range(1, len(history))]

    accuracies = []
    times = []
    for i in iterations:
        model = SupportVectorMachine(loss, i)
        accuracy, t = cross_validation(model, data_array, folds)
        accuracies.append(accuracy)
        times.append(t)

    return values, accuracies, times


def test_on_breast_cancer_dataset():
    measures_num = 50
    step = 1
    folds = 5
    test_part = 0.2

    data = prepare_data("breast-cancer-wisconsin")
    data_array = data.to_numpy()
    np.random.shuffle(data_array)

    iterations = [step * (i + 1) for i in range(measures_num)]
    hinge_res = test_loss("hinge", iterations, folds, data_array, test_part)
    logistic_res = test_loss("logistic", iterations, folds, data_array, test_part)
    quadratic_res = test_loss("quadratic", iterations, folds, data_array, test_part)

    statistics = ["objective function value", "mean accuracy", "mean time (s)"]
    for i in range(len(statistics)):
        create_measures_plot("breast-cancer-wisconsin dataset", "iterations", statistics[i])
        if i == 0:
            plt.ylim([0, 1])
        plt.plot(iterations, hinge_res[i], color="red", label="hinge loss")
        plt.plot(iterations, logistic_res[i], color="green", label="logistic loss")
        plt.plot(iterations, quadratic_res[i], color="blue", label="quadratic loss")
        plt.legend()
        plt.savefig("statistics-" + str(i) + ".png")
        plt.show()


def generate_points():
    file = open("data/custom.data", "w")

    class1 = np.random.normal([-1, 1], [0.8, 0.8], [50, 2])
    class1 = np.append(class1, np.full((50, 1), [-1]), axis=1)
    class2 = np.random.normal([1, -1], [0.8, 0.8], [50, 2])
    class2 = np.append(class2, np.full((50, 1), [1]), axis=1)
    full = np.concatenate((class1, class2), axis=0)
    np.random.shuffle(full)
    for p in full:
        p[0] = np.around(p[0], decimals=3)
        p[1] = np.around(p[1], decimals=3)
        file.write(str(p[0]) + ", " + str(p[1]) + ", " + str(p[2]) + "\n")

    file.close()


def create_features_plot(x_array, y_array):
    for i in range(len(y_array)):
        if y_array[i] == -1:
            plt.scatter(x_array[i][0], x_array[i][1], c="b")
        else:
            plt.scatter(x_array[i][0], x_array[i][1], c="r")


def test_anomalies():
    data = prepare_data("custom")
    data_array = data.to_numpy()
    x = np.delete(data_array, -1, 1)
    y = data_array[:, -1]
    losses = ["hinge", "logistic", "quadratic"]
    colors = ["c", "m", "y"]
    max_points = 3

    for loss in losses:
        model = SupportVectorMachine(LossFunction.by_name(loss), 50)
        model.fit(x, y)
        w0 = model.get_optimizer()

        print(loss + " loss accuracy before: " + str(model.evaluate(x, y)))
        print(loss + " loss optimizer before: " + str(w0))
        init_slope = -w0[0] / w0[1]
        print(loss + " loss separating line slope before: " + str(init_slope) + "\n")

        create_features_plot(x, y)
        plt.axline((0, 0), slope=init_slope, color="g", label="initial")

        for i in range(1, max_points + 1):
            degenerate_x = np.random.normal([10, -5], [2, 2], [i, 2])
            degenerate_y = np.full(len(degenerate_x), 1)
            model.fit(np.concatenate((x, degenerate_x), axis=0), np.concatenate((y, degenerate_y), axis=0))
            w1 = model.get_optimizer()

            print(loss + " loss accuracy after (" + str(i) + " points): " + str(model.evaluate(x, y)))
            print(loss + " loss optimizer after (" + str(i) + " points): " + str(w1))
            slope = -w1[0] / w1[1]
            print(loss + " loss separating line slope after (" + str(i) + " points): " + str(slope))
            print("difference: " + str(abs(slope - init_slope)) + "\n")

            if i == 1:
                label = "1 outlier"
            else:
                label = str(i) + " outliers"
            plt.axline((0, 0), slope=slope, color=colors[i - 1], label=label)

        plt.legend()
        plt.savefig(loss + ".png")
        plt.show()


if __name__ == '__main__':
    test_on_breast_cancer_dataset()
    # test_anomalies()

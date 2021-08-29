import matplotlib.pyplot as plt
import numpy as np


from data_manager import prepare_data, split_data_into_test_train_arrays
from svm import SupportVectorMachine, LossFunction


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


def build_plot(x_array, y_array, w):
    for i in range(len(y_array)):
        if y_array[i] == -1:
            plt.scatter(x_array[i][0], x_array[i][1], c="blue")
        else:
            plt.scatter(x_array[i][0], x_array[i][1], c="red")
    plt.axline((0, 0), slope=-w[0] / w[1], color="green")
    plt.show()


def test_anomalies():
    data = prepare_data("data/custom.data", "custom")
    x_train, y_train, x_test, y_test = split_data_into_test_train_arrays(data, 0.33)

    model = SupportVectorMachine(LossFunction.HINGE)
    w0 = model.fit(x_train, y_train, len(x_train), 1000)
    print(model.evaluate(x_test, y_test))
    build_plot(x_train, y_train, w0)
    build_plot(x_test, y_test, w0)

    degenerate_x = np.random.normal([10, -5], [2, 2], [1, 2])
    degenerate_y = np.full(len(degenerate_x), -1)

    w0 = model.fit(np.concatenate((x_train, degenerate_x), axis=0),
                   np.concatenate((y_train, degenerate_y), axis=0), len(x_train) + len(degenerate_x), 1000)
    print(model.evaluate(x_test, y_test))
    build_plot(x_train, y_train, w0)
    build_plot(x_test, y_test, w0)

    # data = prepare_data("data/breast-cancer-wisconsin.data", "breast-cancer-wisconsin")
    # x_train, y_train, x_test, y_test = split_data_into_test_train_arrays(data, 0.33)
    model = SupportVectorMachine(LossFunction.LOGISTIC)
    w0 = model.fit(x_train, y_train, len(x_train), 1000)
    print(model.evaluate(x_test, y_test))
    build_plot(x_train, y_train, w0)
    build_plot(x_test, y_test, w0)

    w0 = model.fit(np.concatenate((x_train, degenerate_x), axis=0),
                   np.concatenate((y_train, degenerate_y), axis=0), len(x_train) + len(degenerate_x), 1000)
    print(model.evaluate(x_test, y_test))
    build_plot(x_train, y_train, w0)
    build_plot(x_test, y_test, w0)

    # data = prepare_data("data/adult.data", "adult")
    # x_train, y_train, x_test, y_test = split_data_into_test_train_arrays(data, 0.33)
    model = SupportVectorMachine(LossFunction.QUADRATIC)
    w0 = model.fit(x_train, y_train, len(x_train), 1000)
    print(model.evaluate(x_test, y_test))
    build_plot(x_train, y_train, w0)
    build_plot(x_test, y_test, w0)

    w0 = model.fit(np.concatenate((x_train, degenerate_x), axis=0),
                   np.concatenate((y_train, degenerate_y), axis=0), len(x_train) + len(degenerate_x), 1000)
    print(model.evaluate(x_test, y_test))
    build_plot(x_train, y_train, w0)
    build_plot(x_test, y_test, w0)


if __name__ == '__main__':
    test_anomalies()

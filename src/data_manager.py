import math

import numpy as np
import pandas as pd


def prepare_data(filepath, dataset_name):
    if dataset_name == "adult":
        return __prepare_adult(filepath)
    if dataset_name == "breast-cancer-wisconsin":
        return __prepare_breast_cancer(filepath)
    if dataset_name == "custom":
        return __prepare_custom(filepath)

    raise ValueError("invalid dataset")


def __prepare_adult(filepath):
    headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "income"]
    categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                           "native-country"]

    data = pd.read_csv(filepath, sep=r"\s*,\s*", engine="python", names=headers, na_values="?")

    # use one hot encoding
    data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns, dummy_na=True)
    # drop rows with missing values, 30162 rows left
    data = data.dropna(how="any")
    for category in categorical_columns:
        data = data.drop(data[data[category + "_nan"] == 1].index)
    # delete NaN-dummy-columns
    for category in categorical_columns:
        del data[category + "_nan"]

    # make 'income' column the last one
    columns = data.columns.tolist()
    income_column_index = data.columns.get_loc("income")
    columns = columns[:income_column_index] + columns[income_column_index + 1:] + [columns[income_column_index]]
    data = data[columns]

    # replace 'income' strings with integers for classification
    data["income"] = data["income"].replace({">50K": -1, "<=50K": 1})

    return data


def __prepare_breast_cancer(filepath):
    headers = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
               "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
               "Mitoses", "Class"]

    data = pd.read_csv(filepath, sep=r"\s*,\s*", engine="python", names=headers, na_values="?")
    # drop rows with missing values, 683 rows left
    data = data.dropna(how="any")
    data["Class"] = data["Class"].replace({2: -1, 4: 1})
    # drop id column
    data.drop("Sample code number", axis="columns", inplace=True)

    return data


def __prepare_custom(filepath):
    headers = ["1", "2", "class"]

    data = pd.read_csv(filepath, sep=r"\s*,\s*", engine="python", names=headers, na_values="?")

    return data


def split_data_into_test_train_arrays(data, test_part):
    outcome_column_index = len(data.columns) - 1
    data_array = data.to_numpy()
    np.random.shuffle(data_array)
    x = np.delete(data_array, outcome_column_index, 1)
    y = data_array[:, outcome_column_index]

    len_test = math.floor(len(x) * test_part)
    len_train = len(x) - len_test
    x_train, x_test = np.split(x, [len_train])
    y_train, y_test = np.split(y, [len_train])

    return x_train, y_train, x_test, y_test


def encode_categorical_data(data, columns):
    for column in columns:
        data[column] = data[column].astype("category")
        data[column] = data[column].cat.codes

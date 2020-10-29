#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="boston", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data = sklearn.model_selection.train_test_split(dataset.data, test_size = args.test_size, random_state = args.seed)
    integerColumns = []
    otherColumns = []
    for i in range(train_data.shape[1]):
        onlyDigits = True
        for j in range(train_data.shape[0]):
            if not train_data[j,i].is_integer():
                onlyDigits = False
        if onlyDigits == True:
            integerColumns.append(i)
        else:
            otherColumns.append(i)

    integerDataTrain = None
    integerDataTest = None
    for column in integerColumns:
        if integerDataTrain is None:
            integerDataTrain = np.mat(train_data[:,column])
            integerDataTest = np.mat(test_data[:,column])
        else:
            integerDataTrain = np.concatenate((integerDataTrain ,np.mat(train_data[:,column])), axis = 0)
            integerDataTest = np.concatenate((integerDataTest ,np.mat(test_data[:,column])), axis = 0)

    otherDataTrain = None
    otherDataTest = None
    for column in otherColumns:
        if otherDataTrain is None:
            otherDataTrain = np.mat(train_data[:,column])
            otherDataTest = np.mat(test_data[:,column])
        else:
            otherDataTrain = np.concatenate((otherDataTrain ,np.mat(train_data[:,column])), axis = 0)
            otherDataTest = np.concatenate((otherDataTest ,np.mat(test_data[:,column])), axis = 0)


    if(integerDataTrain is not None):
        integerDataTrain = integerDataTrain.T
        integerDataTest = integerDataTest.T

        oneHot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        oneHot.fit(integerDataTrain)
        oneHotEncodedTrain = oneHot.transform(integerDataTrain)
        oneHotEncodedTest = oneHot.transform(integerDataTest)

    if(otherDataTrain is not None):
        otherDataTrain = otherDataTrain.T
        otherDataTest = otherDataTest.T

        scaler = StandardScaler()
        scaler.fit(otherDataTrain)
        scaledTrain = scaler.transform(otherDataTrain)
        scaledTest = scaler.transform(otherDataTest)

    if (integerDataTrain is not None and otherDataTrain is not None):
        train_data = np.concatenate((oneHotEncodedTrain,scaledTrain), axis = 1)
        test_data = np.concatenate((oneHotEncodedTest,scaledTest), axis = 1)
    else:
        if integerDataTrain is not None:
            train_data = oneHotEncodedTrain
            test_data = oneHotEncodedTest
        else:
            train_data = scaledTrain
            test_data = scaledTest

    poly = PolynomialFeatures(2, include_bias=False)
    poly.fit(train_data)
    train_data = poly.transform(train_data)
    test_data = poly.transform(test_data)

    
    return train_data, test_data
'''
    if(integerDataTrain is not None):
        integerDataTrain = integerDataTrain.T
        integerDataTest = integerDataTest.T

        oneHot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        oneHot.fit(integerDataTrain)
        oneHotEncodedTrain = oneHot.transform(integerDataTrain)
        oneHotEncodedTest = oneHot.transform(integerDataTest)

        poly = PolynomialFeatures(2, include_bias=False)
        poly.fit(oneHotEncodedTrain)
        oneHotEncodedTrain = poly.transform(oneHotEncodedTrain)
        oneHotEncodedTest = poly.transform(oneHotEncodedTest)

    if(otherDataTrain is not None):
        otherDataTrain = otherDataTrain.T
        otherDataTest = otherDataTest.T

        scaler = StandardScaler()
        scaler.fit(otherDataTrain)
        scaledTrain = scaler.transform(otherDataTrain)
        scaledTest = scaler.transform(otherDataTest)

        poly = PolynomialFeatures(2, include_bias=False)
        poly.fit(scaledTrain)
        scaledTrain = poly.transform(scaledTrain)
        scaledTest = poly.transform(scaledTest)

    if (integerDataTrain is not None and otherDataTrain is not None):
        train_data = np.concatenate((oneHotEncodedTrain,scaledTrain), axis = 1)
        test_data = np.concatenate((oneHotEncodedTest,scaledTest), axis = 1)
    else:
        if integerDataTrain is not None:
            train_data = oneHotEncodedTrain
            test_data = oneHotEncodedTest
        else:
            train_data = scaledTrain
            test_data = scaledTest
'''
    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    # TODO: Generate polynomial features of order 2 from the current features.
    # If the input values are [a, b, c, d], you should generate
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 60))))

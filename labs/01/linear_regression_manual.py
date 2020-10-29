#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
from sklearn.metrics import mean_squared_error
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    # TODO: Predict target values on the test set

    # TODO: Compute root mean square error on the test set predictions
    rowCount = dataset.data.shape[0]
    onesVector = np.ones((rowCount, 1))
    modifiedData = np.concatenate((dataset.data, onesVector), axis = 1)

    splitData = sklearn.model_selection.train_test_split(modifiedData, dataset.target, test_size = args.test_size, random_state = args.seed)

    trainingData = splitData[0]
    testData = splitData[1]
    trainingTarget = splitData[2]
    testTarget = splitData[3]

    w = np.linalg.inv(trainingData.T @ trainingData) @ trainingData.T @ trainingTarget

    testResult = testData @ w

    rmse = np.sqrt(mean_squared_error(testTarget, testResult))

    return rmse

if __name__ == "__main__":
    args = parser.parse_args()
    rmse = main(args)
    print("{:.2f}".format(rmse))

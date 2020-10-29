#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--range", default=9, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=40, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)
    xs = xs.reshape(-1,1)
    ys = ys.reshape(-1,1)
    rmses = []
    for order in range(1, args.range + 1):
        # TODO: Create features of x^1, ..., x^order.

        if (order > 1):
            newXs = xs[:,0].reshape(-1,1) ** order
            xs = np.concatenate((xs, newXs), axis = 1)

        # TODO: Split the data into a train set and a test set.
        # Use `sklearn.model_selection.train_test_split` method call, passing
        # arguments `test_size=args.test_size, random_state=args.seed`.

        # TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`.

        # TODO: Predict targets on the test set using the trained model.

        # TODO: Compute root mean square error on the test set predictions

        splitData = sklearn.model_selection.train_test_split(xs, ys, test_size = args.test_size, random_state = args.seed)

        trainingData = splitData[0]
        testData = splitData[1]
        trainingTarget = splitData[2]
        testTarget = splitData[3]

        reg = LinearRegression().fit(trainingData, trainingTarget)

        testResult = reg.predict(testData)

        rmse = np.sqrt(mean_squared_error(testResult, testTarget))

        rmses.append(rmse)

    return rmses

if __name__ == "__main__":
    args = parser.parse_args()
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))

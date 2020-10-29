#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.7, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, uncomment the following line.

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size = args.test_size, random_state = args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)

    model = Pipeline(steps = [
        ('scaler', MinMaxScaler()),
        ('poly', PolynomialFeatures()),
        ('logic', LogisticRegression(random_state=args.seed))
    ])

    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    params = {
        'poly__degree': [1,2],
        'logic__C': [0.01, 1, 100],
        'logic__solver': ['lbfgs', 'sag']
    }
    
    skf = StratifiedKFold(5)
    search = GridSearchCV(model, params, cv=skf.split(train_data, train_target))
    search.fit(train_data, train_target)

    prediction = search.predict(test_data)
    total = prediction.shape[0]
    successes = 0
    for i in range(prediction.shape[0]):
        if (prediction[i] == test_target[i]):
            successes += 1

    test_accuracy = successes/total

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))

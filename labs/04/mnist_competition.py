#!/usr/bin/env python3
# 1cef671d-b420-11e7-a937-00505601122b
# 7f104f86-b2ae-11e7-a937-00505601122b 
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
import sklearn.metrics

import numpy as np

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=10, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
        print('\nTransformer init\n')

    def fit(self, X, y = None):
        #print('\nFitt\n')
        return self

    def transform(self, X, y = None):
        #print('\nTransforming\n')
        data = X.copy()

        rowCount = data.shape[0]
        onesVector = np.ones((rowCount, 1))
        data = np.concatenate((data, onesVector), axis = 1)

        '''oneHot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        oneHot.fit(X)
        oneHotEncoded = oneHot.transform(X)'''
        #poly = PolynomialFeatures(2, include_bias=False)

        #poly.fit(data)
        #polyInteger = poly.transform(data)

        #data = np.concatenate((polyInteger), axis = 1)
        return data


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        #train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(train.data, train.target, test_size = 0.2, random_state = args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        model = Pipeline(steps = [
            ('trans', MyTransformer()),
            ('mlp', MLPClassifier(activation='relu', alpha=0.00015, hidden_layer_sizes=(150,)))
        ])

        model.fit(train.data, train.target)
        #res = model.predict(test_data)
        #print(sklearn.metrics.accuracy_score(test_target, res))
        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

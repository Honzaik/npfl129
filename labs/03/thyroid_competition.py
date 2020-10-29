#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


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


        integerPart = data[:,0:14]
        floatPart = data[:,15:]
        '''
        oneHot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        oneHot.fit(integerPart)
        oneHotEncoded = oneHot.transform(integerPart)

        scaler = StandardScaler()
        scaler.fit(floatPart)
        scaledTrain = scaler.transform(floatPart)

        poly = PolynomialFeatures(2, include_bias=False)

        poly.fit(oneHotEncoded)
        polyInteger = poly.transform(oneHotEncoded)


        poly = PolynomialFeatures(2, include_bias=False)

        poly.fit(scaledTrain)
        polyFloat = poly.transform(scaledTrain)

        #combined = np.concatenate((oneHotEncoded,scaledTrain), axis = 1)

        data = np.concatenate((polyInteger,polyFloat), axis = 1)
        '''

        scaler = MinMaxScaler()
        scaler.fit(floatPart)
        scaledTrain = scaler.transform(floatPart)

        poly = PolynomialFeatures(3, include_bias=False)

        poly.fit(integerPart)
        polyInteger = poly.transform(integerPart)

        poly = PolynomialFeatures(3, include_bias=False)

        poly.fit(scaledTrain)
        polyFloat = poly.transform(scaledTrain)

        data = np.concatenate((polyInteger,polyFloat), axis = 1)
        return data


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=33, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        #train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(train.data, train.target, test_size = 0.5, random_state = args.seed)
        train_data, train_target = train.data, train.target

        model = Pipeline(steps = [
            ('trans', MyTransformer()),
            ('logic', LogisticRegression(C = 100, solver="lbfgs", random_state=args.seed, max_iter=2000))
        ])
        params = {
            'poly__degree': [2,3],
            'logic__C': [0.01, 1, 100],
            'logic__solver': ['lbfgs', 'sag']
        }
        '''
        skf = StratifiedKFold(5)
        search = GridSearchCV(model, params, cv=skf.split(train_data, train_target))
        search.fit(train_data, train_target)

        print(search.best_params_)
        prediction = search.predict(test_data)
        total = prediction.shape[0]
        successes = 0
        for i in range(prediction.shape[0]):
            if (prediction[i] == test_target[i]):
                successes += 1

        test_accuracy = successes/total

        print(test_accuracy)
        '''
        model.fit(train_data, train_target)
        '''
        prediction = model.predict(test_data)
        total = prediction.shape[0]
        successes = 0
        for i in range(prediction.shape[0]):
            if (prediction[i] == test_target[i]):
                successes += 1

        test_accuracy = successes/total

        print(test_accuracy)
        '''
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

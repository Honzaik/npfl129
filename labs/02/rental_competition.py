#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import numpy as np

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: sprint, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
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

        integerPart = data[:,0:7]
        floatPart = data[:,8:]

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

        combined = np.concatenate((oneHotEncoded,scaledTrain), axis = 1)

        data = np.concatenate((polyInteger,polyFloat,combined), axis = 1)
        return data

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        #trainData, devData, trainTarget, devTarget = sklearn.model_selection.train_test_split(train.data, train.target, test_size = 0.0, random_state = args.seed)
        # TODO: Train a model on the given dataset and store it in `model`.
        #for l in np.arange(0, 10, 0.1):
        l = 2.6
        model = Pipeline(steps = [
            ('trans', MyTransformer()),
            ('model', Ridge(l))
        ])

        model.fit(train.data, train.target)

        #res = model.predict(devData)
        #rmse = np.sqrt(mean_squared_error(res, devTarget))
        #print(str(round(l,2)) + ' ' + str(rmse))

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

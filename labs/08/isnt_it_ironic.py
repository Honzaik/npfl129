#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection
from sklearn.naive_bayes import MultinomialNB

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(os.path.basename(name).replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_data, train_target = train.data, train.target
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(train.data, train.target, test_size = 0.2, random_state = args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        model = Pipeline(steps = [
            ('trans',TfidfVectorizer()),
            ('baive', MultinomialNB())
        ])

        model.fit(train_data, train_target)

        #predictions = model.predict(test_data);

        #print(f1_score(test_target, predictions));

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list of a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

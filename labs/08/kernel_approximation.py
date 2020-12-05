#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.svm

class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=0.022, type=float, help="RBF gamma")
parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
parser.add_argument("--nystroem", default=300, type=int, help="Use Nystroem approximation")
parser.add_argument("--original", default=False, action="store_true", help="Use original data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--rff", default=0, type=int, help="Use RFFs")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--svm", default=False, action="store_true", help="Use SVM instead of LR")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class RFFsTransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    ws = []
    bs = []

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # TODO: Generate suitable `w` and `b`.
        # To obtain deterministic results, generate
        # - `w` first, using a single `generator.normal` call with
        #   output shape `(input_features, self._n_components)`
        # - `b` second, using a single `generator.uniform` call
        #   with output shape `(self._n_components,)`

        #self.ws = np.sqrt(2*self._gamma) * generator.normal(0, 1, size=(self._n_components, X.shape[1]))
        self.ws = np.sqrt(2*self._gamma) * generator.normal(0, 1, size=(X.shape[1], self._n_components))
        self.bs = generator.uniform(0, 2*np.pi, size=(self._n_components))
        return self

    def getPhi(self, x, index):
        result = np.sqrt(2 / self._n_components) * np.cos(self.ws[index] @ x + self.bs[index])
        return result

    def transform(self, X):
        newX = []
        for i in range(X.shape[0]):
            newRow = []
            newRow = np.sqrt(2 / self._n_components) * np.cos(self.ws.T @ X[i] + self.bs)
            #for j in range(self._n_components):
                #newRow.append(self.getPhi(X[i], j))
            newRow = np.array(newRow)
            newX.append(newRow)

        newX = np.array(newX)


        return newX

class NystroemTransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def _rbf_kernel(self, X, Z):
        # TODO: Compute the RBF kernel with `self._gamma` for
        # given two sets of examples.
        #
        # A reasonably efficient implementation should probably compute the
        # kernel line-by-line, computing K(X_i, Z) using a single `np.linalg.norm`
        # call, and then concatenate the results using `np.stack`.
        kernelLines = []
        for i in range(X.shape[0]):
            XiRepeat = np.tile(X[i], Z.shape[0]).reshape(Z.shape[0], X[i].shape[0])
            norm = np.linalg.norm(XiRepeat - Z, axis=1)
            powerNorm = np.power(norm, 2)

            res = np.exp(- self._gamma * powerNorm)
            
            kernelLines.append(res)

        kernelLines = np.array(kernelLines)

        return kernelLines

    K = []
    V = []
    chosen = []

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # TODO: Choose a random subset of examples, utilizing indices
        #   indices = generator.choice(X.shape[0], size=self._n_components, replace=False)
        #
        # Then, compute K as the RBF kernel of the chosen examples and
        # V as K^{-1/2} -- use `np.linalg.svd(K, hermitian=True)` to compute
        # the SVD (equal to eigenvalue decomposition for real symmetric matrices).
        # Add 1e-12 to the diagonal matrix returned by SVD before computing
        # the inverse of the square root.
        indices = generator.choice(X.shape[0], size=self._n_components, replace=False)

        self.chosen = X[indices]
        #self.K = self._getKMatrix(self.chosen, self.chosen)
        self.K = self._rbf_kernel(self.chosen, self.chosen)

        u, s, v = np.linalg.svd(self.K, hermitian=True)
        s += 1e-12
        s = s**(-1/2)
        diag = np.diag(s)

        self.V = u @ diag @ v

        return self

    def transform(self, X):
        # TODO: Compute the RBF kernel of `X` and the chosen training examples
        # and then process it using the precomputed `V`.
        kernel = self._rbf_kernel(X, self.chosen)
        newX = []
        for i in range(X.shape[0]):
            newX.append(np.array(self.V.T @ kernel[i]))

        return np.array(newX)

def main(args):
    # Use the digits dataset.
    dataset = MNIST(data_size=5000)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    features = []
    if args.original:
        features.append(("original", sklearn.preprocessing.FunctionTransformer()))
    if args.rff:
        features.append(("rff", RFFsTransformer(args.rff, args.gamma, args.seed)))
    if args.nystroem:
        features.append(("nystroem", NystroemTransformer(args.nystroem, args.gamma, args.seed)))

    if args.svm:
        classifier = sklearn.svm.SVC()
    else:
        classifier = sklearn.linear_model.LogisticRegression(solver="saga", penalty="none", max_iter=args.max_iter, random_state=args.seed)

    pipeline = sklearn.pipeline.Pipeline([
        ("scaling", sklearn.preprocessing.MinMaxScaler()),
        ("features", sklearn.pipeline.FeatureUnion(features)),
        ("classifier", classifier),
    ])
    pipeline.fit(train_data, train_target)

    test_accuracy = sklearn.metrics.accuracy_score(test_target, pipeline.predict(test_data))
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))

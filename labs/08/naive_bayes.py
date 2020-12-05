#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
from sklearn.metrics import accuracy_score
from scipy.stats import norm

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=10, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values

def gauss(x, mean, variance):
    coef = 1/np.sqrt(2*np.pi*variance)
    value = np.exp((-1)*( ((x-mean)**2) / (2*variance)))
    return coef*value

def getProbabilitiesMatrix(args, data, target, classes):
    
    if args.naive_bayes_type == 'gaussian':
        matrix = np.zeros(shape=(data.shape[1], len(classes), 2))
        for c in classes:
            for i in range(data.shape[1]):
                mean = 0
                count = 0
                for j in range(data.shape[0]):
                    if target[j] == c:
                        mean += data[j][i]
                        count += 1
                mean /= count 
                variance = 0
                for j in range(data.shape[0]):
                    if target[j] == c:
                        variance += (data[j][i] - mean)**2
                variance /= count
                variance += args.alpha
                matrix[i][c] = [mean, variance]
    elif args.naive_bayes_type == 'multinomial':
        matrix = np.zeros(shape=(data.shape[1], len(classes), 1))
        for c in range(len(classes)):
            for i in range(data.shape[1]):
                nik = 0
                for j in range(data.shape[0]):
                    if target[j] == c:
                        nik += data[j][i]
                nik += args.alpha

                total = 0
                for j in range(data.shape[0]):
                    if target[j] == c:
                        total += np.sum(data[j])

                matrix[i][c] = nik / (total + args.alpha*data.shape[1])
    elif args.naive_bayes_type == 'bernoulli':
        matrix = np.zeros(shape=(data.shape[1], len(classes), 1))
        for c in range(len(classes)):
            for i in range(data.shape[1]):
                count = 0
                total = 0
                for j in range(data.shape[0]):
                    if target[j] == c:
                        if data[j][i] != 0:
                            count += 1
                        total += 1
                count += args.alpha
                matrix[i][c] = count / (total + args.alpha*2)
    return matrix

def main(args):
    # Use the digits dataset.

    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    #p(Ck)
    classes = {}
    for t in train_target:
        if t not in classes:
            classes[t] = 0

        classes[t] += 1

    for c in classes:
        classes[c] /= train_target.shape[0]

    matrix = getProbabilitiesMatrix(args, train_data, train_target, classes)
    predictions = []
    if args.naive_bayes_type == 'gaussian':
        classesProbabilities = {}
        
        for c in classes:
            classesProbabilities[c] = []
            for i in range(test_data.shape[1]):
                probabilities = (norm.pdf(test_data[:,i], matrix[i][c][0], np.sqrt(matrix[i][c][1])))
                classesProbabilities[c].append(np.array(probabilities))
            classesProbabilities[c] = np.array(classesProbabilities[c])

        for j in range(test_data.shape[0]):
            classProbabilities = np.zeros(len(classes))
            for c in classes:
                prob = classes[c]
                for i in range(test_data.shape[1]):
                    prob *= classesProbabilities[c][i][j]
                classProbabilities[c] = prob

            predictions.append(np.argmax(classProbabilities))
        
        '''
        for j in range(test_data.shape[0]):
            classProbabilities = np.zeros(len(classes))
            for c in classes:
                prob = classes[c]
                for i in range(test_data.shape[1]):
                    prob *= gauss(test_data[j][i], matrix[i][c][0], matrix[i][c][1]) #norm.pdf(test_data[j][i], matrix[i][c][0], matrix[i][c][1])
                classProbabilities[c] = prob
            predictions.append(np.argmax(classProbabilities))
        '''
    elif args.naive_bayes_type == 'multinomial':
        for j in range(test_data.shape[0]):
            classProbabilities = np.zeros(len(classes))
            for c in range(len(classes)):
                prob = classes[c]
                for i in range(test_data.shape[1]):
                    prob *= (matrix[i][c]**(test_data[j][i]))*10**6
                classProbabilities[c] = prob

            predictions.append(np.argmax(classProbabilities))

    elif args.naive_bayes_type == 'bernoulli':
        for j in range(test_data.shape[0]):
            classProbabilities = np.zeros(len(classes))
            for c in range(len(classes)):
                prob = classes[c]
                for i in range(test_data.shape[1]):
                    xi = 0
                    if (test_data[j][i] != 0):
                        xi = 1
                    prob *= (matrix[i][c]**xi)*((1-matrix[i][c])**(1-xi))
                classProbabilities[c] = prob

            predictions.append(np.argmax(classProbabilities))


    # TODO: Fit the naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Do not forget that Bernoulli NB works with binary data, so consider
    #   all non-zero features as ones during both estimation and prediction.

    # TODO: Predict the test data classes and compute test accuracy.
    test_accuracy = accuracy_score(test_target, predictions)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))

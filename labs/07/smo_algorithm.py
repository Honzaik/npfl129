#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from numpy import linalg
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args, x, y):
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    if args.kernel == 'poly':
        return (args.kernel_gamma * (x.T @ y) + 1) ** args.kernel_degree
    else:
        norm = np.sum(np.power(x-y, 2))
        return np.exp(-args.kernel_gamma * norm)

def getKMatrix(args, data):
    matrix = np.zeros((data.shape[0], data.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            matrix[i][j] = kernel(args, data[i], data[j])
    return matrix

def getY(xIndex, a, target, matrix, b):
    y = 0
    for k in range(len(a)): # calculates y(x_i)
        y += a[k]*target[k]*matrix[k][xIndex]
    y+= b
    return y

def getPredictions(args, a, b, trainData, trainTarget, toPredict):
    ys = np.zeros(toPredict.shape[0])
    for i in range(toPredict.shape[0]):
        value = 0
        for j in range(len(a)):
            value += a[j] * trainTarget[j] * kernel(args, trainData[j], toPredict[i])
        value += b
        if value > 0:
            ys[i] = 1
        else:
            ys[i] = -1

    return ys

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, train_data, train_target, test_data, test_target):
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)
    trainMatrix = getKMatrix(args, train_data)
    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)
            t_i = train_target[i]
            y_i = getY(i, a, train_target, trainMatrix, b)
            E_i = y_i - t_i

            # TODO: Check that a[i] fulfuls the KKT condition, using `args.tolerance` during comparisons.
            # If the conditions, do not hold, then
            # - compute the updated unclipped a_j^new.
            if ((a[i] < args.C - args.tolerance) and (t_i*E_i < -args.tolerance)) or ((a[i] > args.tolerance) and (t_i*E_i > args.tolerance)):
                t_j = train_target[j]
                y_j = getY(j, a, train_target, trainMatrix, b)
                E_j = y_j - t_j
                secondDerivative = 2 * trainMatrix[i][j] - trainMatrix[i][i] - trainMatrix[j][j]

                #   If the second derivative of the loss with respect to a[j]
                #   is > -`args.tolerance`, do not update a[j] and continue
                #   with next i.
                if (secondDerivative > -args.tolerance):
                    continue

                a_jnew = a[j] - t_j * ((E_i - E_j) / secondDerivative)
                # - clip the a_j^new to suitable [L, H].
                if t_i == t_j:
                    L = np.max([0, a[i] + a[j] - args.C])
                    H = np.min([args.C, a[i] + a[j]])
                else:
                    L = np.max([0, a[j] - a[i]])
                    H = np.min([args.C, args.C + a[j] - a[i]])

                if a_jnew < L:
                    a_jnew = L
                else:
                    if a_jnew > H:
                        a_jnew = H

                #   If the clipped updated a_j^new differs from the original a[j]
                #   by less than `args.tolerance`, do not update a[j] and continue
                #   with next i.
                if np.abs(a_jnew - a[j]) < args.tolerance:
                    continue

                # - update a[j] to a_j^new, and compute the updated a[i] and b.
                a_inew = a[i] - t_i*t_j*(a_jnew - a[j])

                b_j = b - E_j - t_i*(a_inew - a[i])*trainMatrix[i][j] - t_j*(a_jnew - a[j])*trainMatrix[j][j]
                b_i = b - E_i - t_i*(a_inew - a[i])*trainMatrix[i][i] - t_j*(a_jnew - a[j])*trainMatrix[j][i]

                #   During the update of b, compare the a[i] and a[j] to zero by
                #   `> args.tolerance` and to C using `< args.C - args.tolerance`.
                if (args.tolerance < a_inew and a_inew < args.C - args.tolerance):
                    b = b_i
                elif (args.tolerance < a_jnew and a_jnew < args.C - args.tolerance):
                    b = b_j
                else:
                    b = (b_i+b_j)/2
                
                a[i] = a_inew
                a[j] = a_jnew
                as_changed += 1 

        # TODO: After each iteration, measure the accuracy for both the
        # train test and the test set and append it to `train_accs` and `test_accs`.
        trainPredictions = getPredictions(args, a, b, train_data, train_target, train_data)
        testPredictions = getPredictions(args, a, b, train_data, train_target, test_data)
        train_accs.append(accuracy_score(train_target, trainPredictions))
        test_accs.append(accuracy_score(test_target, testPredictions))

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    print("Training finished after iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors, support_vector_weights = [], []

    for i in range(len(a)):
        if a[i] > args.tolerance:
            support_vectors.append(train_data[i])
            support_vector_weights.append(a[i]*train_target[i])

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args):
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM prediction `y(x)` for the given x.
        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--iterations", default=1, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def softmax(vec):
    maximum = vec.max()
    vec = vec - maximum
    denom = 0
    result = []
    for value in vec:
        denom += np.exp(value)

    for value in vec:
        result.append(np.exp(value)/denom)

    return np.array(result)

def getAccLoss(data, target, weights):
    total = data.shape[0]
    successCount = 0
    lossTotal = 0
    for i in range(data.shape[0]):
        prob = softmax(data[i] @ weights)
        res = np.argmax(prob)
        lossCoef = prob[target[i]]

        if target[i] == res:
            successCount += 1
        else:
            lossCoef = 1 - lossCoef

        lossTotal -= np.log(lossCoef)

    return successCount/total, lossTotal/total

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        temp = inputs @ weights[0] + biases[0]
        hidden = np.maximum(temp, np.zeros(temp.shape))

        output = softmax(hidden @ weights[1] + biases[0])

        return hidden, output

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        rounds = train_data.shape[0]//args.batch_size
        rounds = 1
        for j in range(rounds):
            gradients = [np.zeros(weights[0].shape), np.zeros(weights[1].shape)]
            biases = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]
            for i in range(args.batch_size):
                x_i = train_data[permutation[j*args.batch_size + i]]
                t_i = train_target[permutation[j*args.batch_size + i]]

                h = x_i @ weights[0] + biases[0] # W^h x + b^h
                k = weights[1].shape[1]
                y_in = h @ weights[1] + biases[1]
                temp = h @ weights[1] # W^y h
                softTemp = softmax(y_in)
                dydyin = np.zeros((k,k))
                for ii in range(k):
                    for jj in range(k):
                        dydyin[ii][jj] = -softTemp[ii]*softTemp[jj]
                        if ii == jj:
                            dydyin[ii][jj] += softTemp[ii]
                        
                temp = temp @ dydyin
                for i in range(k):
                    temp[i] = -1/temp[i]
                
                hArray = np.repeat(h, args.classes).reshape(h.shape[0], args.classes)
                gradients[1] += hArray @ np.diag(temp)

                temp = np.ones(k)
                softTemp = softmax(y_in)
                dydyin = np.zeros((k,k))
                for ii in range(k):
                    for jj in range(k):
                        dydyin[ii][jj] = -softTemp[ii]*softTemp[jj]
                        if ii == jj:
                            dydyin[ii][jj] += softTemp[ii]
                        
                temp = temp @ dydyin
                for i in range(k):
                    temp[i] = -1/temp[i]
                
                hArray = np.repeat(h, args.classes).reshape(h.shape[0], args.classes)
                biases[1] += np.ones(k)*temp
                '''
                a = softmax(x_i.T @ weights)
                coef = -a
                coef[t_i] = 1+coef[t_i]
                xArray = np.repeat(x_i, 10).reshape(x_i.shape[0], 10)
                gradient += xArray @ np.diag(coef)
                '''

            #weights = weights + args.learning_rate*(gradient/args.batch_size)

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        train_accuracy, train_loss = 0,0#getAccLoss(train_data, train_target, weights)
        test_accuracy, test_loss = 0,0#getAccLoss(test_data, test_target, weights)

        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")

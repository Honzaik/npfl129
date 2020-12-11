#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrapping", default=True, action="store_true", help="Perform data bootstrapping")
parser.add_argument("--feature_subsampling", default=0.5, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=2, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=46, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=4, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

class Node:
    def __init__(self, indices, data, target, args, depth, generator):
        self.left = None
        self.right = None
        self.indices = indices
        self.data = data
        self.target = target
        self.splittingFeatureIndex = None
        self.splittingFeatureValue = None
        self.classes = np.unique(target)
        self.args = args
        self.depth = depth
        self.prediction = self.getPrediction()
        self.generator = generator

    def getPrediction(self):
        classCounts = []
        for c in self.classes:
            classCounts.append(0)
            for index in self.indices:
                if self.target[index] == c:
                    classCounts[c] += 1
        classCounts = np.array(classCounts)
        return np.argmax(classCounts)

    def isLeaf(self):
        return (self.left == None and self.right == None)

    def getNumberOfIndices(self):
        return len(self.indices)

    def getLeaves(self):
        if (self.isLeaf()):
            return [self]

        leaves = []
        if self.left != None:
            leaves += self.left.getLeaves()
        if self.right != None:
            leaves += self.right.getLeaves()
        return leaves

    def getIndicesBySplitPoint(self, featureIndex, splitPoint):
        indicesLeft = []
        indicesRight = []
        for index in self.indices:
            if (self.data[index][featureIndex] <= splitPoint):
                indicesLeft.append(index)
            else:
                indicesRight.append(index)
        return indicesLeft, indicesRight

    def getCriterionValue(self, indices):
        if indices == None:
            indices = self.indices

        criterionValue = 0
        for c in self.classes:
            prob = 0
            count = 0
            for index in indices:
                if self.target[index] == c:
                    count += 1
            prob = count / len(indices)
            if prob != 0:
                criterionValue += (prob * np.log(prob))
        criterionValue *= (-1)*len(indices)
        return criterionValue

    def predictValue(self, x):
        if self.isLeaf():
            return self.prediction
        else:
            if (x[self.splittingFeatureIndex] <= self.splittingFeatureValue):
                return self.left.predictValue(x)
            else:
                return self.right.predictValue(x)

    def printTree(self):
        if self.isLeaf():
            print(('                   ' * self.depth) + str(self.depth), len(self.indices), round(self.getCriterionValue(None), 3), self.prediction)
        else:
            self.left.printTree()
            print(('                   ' * self.depth) + str(self.depth), self.splittingFeatureIndex, self.splittingFeatureValue, len(self.indices), round(self.getCriterionValue(None), 3), self.prediction)
            self.right.printTree()


def splitNode(nodeToSplit, save):

    if (nodeToSplit.args.max_depth != None) and (nodeToSplit.depth >= nodeToSplit.args.max_depth):
        return False, None
    if (len(nodeToSplit.indices) <= 1):
        return False, None
    numberOfFeatures = nodeToSplit.data.shape[1]
    possibleSplitPoints = []
    featureMask = nodeToSplit.generator.uniform(size=numberOfFeatures) <= nodeToSplit.args.feature_subsampling

    for featureIndex in range(numberOfFeatures):
        possibleSplitPoints.append([])
        if not featureMask[featureIndex]:
            continue
        uniqueFeatureValues = np.unique(nodeToSplit.data[nodeToSplit.indices, featureIndex])
        numberOfUniqueFeatureValues = uniqueFeatureValues.shape[0]
        for i in range(numberOfUniqueFeatureValues-1):
            possibleSplitPoints[featureIndex].append(round(((uniqueFeatureValues[i+1] + uniqueFeatureValues[i]) / 2), 5))


    totalCriterion = nodeToSplit.getCriterionValue(None)
    smallestCriterionValue = np.Inf
    smallestCriterion = []
    smallestIndices = [[],[]]

    for featureIndex in range(numberOfFeatures):
        if not featureMask[featureIndex]:
            continue
        splitPoints = possibleSplitPoints[featureIndex]
        for splitPoint in splitPoints:
            leftIndices, rightIndices = nodeToSplit.getIndicesBySplitPoint(featureIndex, splitPoint)
            leftCriterion =  nodeToSplit.getCriterionValue(leftIndices)
            rightCriterion =  nodeToSplit.getCriterionValue(rightIndices)
            value = leftCriterion + rightCriterion - totalCriterion
            if (value < smallestCriterionValue):
                smallestCriterionValue = value
                smallestCriterion = [featureIndex, splitPoint]
                smallestIndices = [leftIndices, rightIndices]

    if (smallestCriterionValue == 0 or len(smallestIndices[0]) == 0 or len(smallestIndices[1]) == 0) :
        return False, None

    if (save == True):
        leftNode = Node(smallestIndices[0], nodeToSplit.data, nodeToSplit.target, nodeToSplit.args, nodeToSplit.depth + 1, nodeToSplit.generator)
        rightNode = Node(smallestIndices[1], nodeToSplit.data, nodeToSplit.target, nodeToSplit.args, nodeToSplit.depth + 1, nodeToSplit.generator)
        nodeToSplit.splittingFeatureIndex = smallestCriterion[0]
        nodeToSplit.splittingFeatureValue = smallestCriterion[1]
        nodeToSplit.left = leftNode
        nodeToSplit.right = rightNode

    return True, smallestCriterionValue

def splitDepth(node):
    if (node.isLeaf()):
        success, value = splitNode(node, True)
        if (success):
            splitDepth(node.left)
            splitDepth(node.right)

def predict(forest, data):
    predictions = []
    for i in range(data.shape[0]):
        prediction = [0] * len(forest[0].classes)
        for tree in forest:
            treePrediction = tree.predictValue(data[i])
            prediction[treePrediction] += 1
        predictions.append(np.argmax(prediction))
        #print(prediction)
        #print(np.argmax(prediction))

    return predictions



def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    generator = np.random.RandomState(args.seed)

    indices = []
    for i in range(train_data.shape[0]):
        indices.append(i)

    forest = []
    for _ in range(args.trees):
        if args.bootstrapping != False:
            indices = generator.choice(len(train_data), size=len(train_data))
        tree = Node(indices, train_data, train_target, args, 0, generator)
        splitDepth(tree)
        forest.append(tree)

    trainPredictions = predict(forest, train_data)
    testPredictions = predict(forest, test_data)


    # TODO: Create a random forest on the trainining data.
    #
    # For determinism, create a generator
    #   generator = np.random.RandomState(args.seed)
    # at the beginning and then use this instance for all random number generation.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, so split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targetsare not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in left subtrees before nodes in right subtrees.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. When splitting a node, start by generating
    #   a feature mask using
    #     generator.uniform(size=number_of_features) <= feature_subsampling
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not.
    #   (When feature_subsampling == 1, all features are used, but the mask
    #   should still be generated.)
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bootstrapping` is set, right before training a decision tree,
    #   create a bootstrap sample of the training data using the following indices
    #     indices = generator.choice(len(train_data), size=len(train_data))
    #   and if `args.bootstrapping` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with smallest class index in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = accuracy_score(train_target, trainPredictions)
    test_accuracy = accuracy_score(test_target, testPredictions)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))

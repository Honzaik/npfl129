#!/usr/bin/env python3
# 1cef671d-b420-11e7-a937-00505601122b
# 7f104f86-b2ae-11e7-a937-00505601122b 
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="entropy", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=6, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=40, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=97, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class Node:
    def __init__(self, indices, data, target, args, depth):
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
        if self.args.criterion == 'gini':
            for c in self.classes:
                count = 0
                for index in indices:
                    if self.target[index] == c:
                        count += 1
                prob = count / len(indices)
                criterionValue += (prob * (1 - prob))
            criterionValue *= len(indices)
        else:
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

    if (len(nodeToSplit.indices) < nodeToSplit.args.min_to_split):
        return False, None

    numberOfFeatures = nodeToSplit.data.shape[1]
    possibleSplitPoints = []
    for featureIndex in range(numberOfFeatures):
        uniqueFeatureValues = np.unique(nodeToSplit.data[nodeToSplit.indices, featureIndex])
        smallestDistance = np.Inf
        possibleSplitPoints.append([])
        numberOfUniqueFeatureValues = uniqueFeatureValues.shape[0]
        for i in range(numberOfUniqueFeatureValues-1):
            possibleSplitPoints[featureIndex].append(round(((uniqueFeatureValues[i+1] + uniqueFeatureValues[i]) / 2), 5))


    totalCriterion = nodeToSplit.getCriterionValue(None)
    smallestCriterionValue = np.Inf
    smallestCriterion = []
    smallestIndices = []

    for featureIndex in range(numberOfFeatures):
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

    if (smallestCriterionValue == 0):
        return False, None

    if (save == True):
        leftNode = Node(smallestIndices[0], nodeToSplit.data, nodeToSplit.target, nodeToSplit.args, nodeToSplit.depth + 1)
        rightNode = Node(smallestIndices[1], nodeToSplit.data, nodeToSplit.target, nodeToSplit.args, nodeToSplit.depth + 1)
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

def predict(tree, data):
    predictions = []
    for i in range(data.shape[0]):
        predictions.append(tree.predictValue(data[i]))

    return predictions

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    indices = []
    for i in range(train_data.shape[0]):
        indices.append(i)

    tree = Node(indices, train_data, train_target, args, 0)

    if args.max_leaves == None:
        splitDepth(tree)
    else:
        while (len(tree.getLeaves()) < args.max_leaves):
            leaves = tree.getLeaves()
            smallestCriterion = np.Inf
            bestLeaf = None
            for leaf in leaves:
                success, value = splitNode(leaf, False)
                if success and value < smallestCriterion:
                    smallestCriterion = value
                    bestLeaf = leaf
            if bestLeaf == None:
                break
            splitNode(bestLeaf, True)

    #tree.printTree()

    trainPredictions = predict(tree, train_data)
    testPredictions = predict(tree, test_data)

    # TODO: Create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = accuracy_score(train_target, trainPredictions)
    test_accuracy = accuracy_score(test_target, testPredictions)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))

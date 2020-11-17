#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
import sklearn.metrics

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default='ss', type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization_tanh_500.model", type=str, help="Model path")

def makeVectors(text, diaText, size):
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    LETTERS_NODIA = "acdeeinorstuuyz"

    mapp = {
    'a' : ['a', 'á'],
    'c' : ['c', 'č'],
    'd' : ['d', 'ď'],
    'e' : ['e', 'é', 'ě'],
    'i' : ['i', 'í'],
    'n' : ['n', 'ň'],
    'o' : ['o', 'ó'],
    'r' : ['r', 'ř'],
    's' : ['s', 'š'],
    'u' : ['u', 'ú', 'ů'],
    'y' : ['y', 'ý'],
    'z' : ['z', 'ž'],
    }
    text = text.lower()
    text = text.replace('\n', ' ');
    text = text.replace(':', ' ')
    text = text.replace('"', ' ')
    text = text.replace(':', ' ')
    text = text.replace('?', ' ')
    text = text.replace(',', ' ')
    text = text.replace('-', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')

    diaText = diaText.lower()
    diaText = diaText.replace('\n', ' ');
    diaText = diaText.replace(':', ' ')
    diaText = diaText.replace('"', ' ')
    diaText = diaText.replace(':', ' ')
    diaText = diaText.replace('?', ' ')
    diaText = diaText.replace(',', ' ')
    diaText = diaText.replace('-', ' ')
    diaText = diaText.replace('  ', ' ')
    diaText = diaText.replace('  ', ' ')
    
    trainVectors = []
    targets = []

    for m in mapp:
        trainVectors.append([])
        targets.append([])

    #text = text[0:100]
    #diaText = diaText[0:100]

    words = text.split(' ')
    diaWords = diaText.split(' ')
    maxLen = 0
    for word in words:
        if len(word) > maxLen:
            maxLen = len(word)

    vectorLen = maxLen*2
    print('vec len = ' + str(vectorLen))
    centerIndex = vectorLen // 2

    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j] in mapp.keys():
                centeredWord = list(' ' * vectorLen)
                centeredWord[centerIndex-j:centerIndex-j+len(words[i])] = list(words[i])

                index = list(mapp.keys()).index(words[i][j])

                ll = list(map(lambda x: ord(x), centeredWord))

                trainVector = np.array(ll)
                trainVectors[index].append(trainVector)

                clas = mapp[words[i][j]].index(diaWords[i][j])

                targets[index].append(clas)

    trainVectors = np.array(trainVectors)
    targets = np.array(targets)

    for i in range(len(mapp.keys())):
        trainVectors[i] = np.array(trainVectors[i])
        targets[i] = np.array(targets[i])


    return trainVectors, targets

def vectorize(centerCharIndex, word, vectorLen):
    centerIndex = vectorLen // 2
    centeredWord = list(' ' * vectorLen)
    centeredWord[centerIndex-centerCharIndex:centerIndex-centerCharIndex+len(word)] = list(word)


    ll = list(map(lambda x: ord(x), centeredWord))


    return np.array(ll)

'''
def makeVectors(text, diaText, margin):
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    LETTERS_NODIA = "acdeeinorstuuyz"
    listDia = list(LETTERS_DIA)
    listNoDia = list(LETTERS_NODIA)

    text = text.replace('\n', ' ');
    text = text.replace(':', ' ')
    text = text.replace('"', ' ')
    text = text.replace(':', ' ')
    text = text.replace('?', ' ')

    diaText = diaText.replace('\n', ' ');
    diaText = diaText.replace(':', ' ')
    diaText = diaText.replace('"', ' ')
    diaText = diaText.replace(':', ' ')
    diaText = diaText.replace('?', ' ')
    
    trainVectors = []
    targets = []

    for c in listNoDia:
        trainVectors.append([])
        targets.append([])

    ii = 0
    for i in range(len(text)):
        if text[i] in listNoDia:
            startIndex = i - margin
            endIndex = i + margin + 1
            paddingLeft = 0
            paddingRight = 0
            if (startIndex < 0):
                paddingLeft = startIndex * -1
                startIndex = 0
            if (endIndex > len(text)):
                paddingRight = endIndex - len(text)
                endIndex = len(text)

            subtext = (' ' * paddingLeft) + text[startIndex:i] + text[i:endIndex] + (' ' * paddingRight)

            index = LETTERS_NODIA.index(text[i])

            trainVector = np.array(list(map(lambda x: ord(x), list(subtext))))

            trainVectors[index].append(trainVector)
            isDia = 0
            if (diaText[i] != text[i]):
                isDia = 1
            targets[index].append(isDia)

    trainVectors = np.array(trainVectors)
    targets = np.array(targets)

    for i in range(len(listDia)):
        trainVectors[i] = np.array(trainVectors[i])
        targets[i] = np.array(targets[i])


    return trainVectors, targets
'''
def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        trainVectors, targets = makeVectors(train.data, train.target, 3)

        testVector = np.array([vectorize(6, 'sedmnact', 40), vectorize(4, 'limeckem', 40), vectorize(2, 'reci', 40), vectorize(0, 'chladnokrevne', 40)]);
        print(trainVectors.shape[0])
        models = []

        for i in range(trainVectors.shape[0]):
            model = Pipeline(steps = [
                ('trans', MinMaxScaler()),
                ('mlp', MLPClassifier(activation='tanh', hidden_layer_sizes=(300,), max_iter=400))
            ])
            model.fit(trainVectors[i], targets[i])
            models.append(model)
            print(i, 'done')

        '''

        scaler = MinMaxScaler()
        scaledTrain = scaler.fit_transform(trainVectors[1])
        scaledTest = scaler.transform(testVector)
        mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(200,), max_iter=300)
        mlp.fit(scaledTrain, targets[1])
        print(mlp.classes_)
        probs = mlp.predict_proba(scaledTest)
        weights = np.array([1.469, 3.131])
        for prob in probs:
            print(prob)
            print(np.argmax(prob * weights))


        

        testVector = np.array([vectorize(5, 'naramek', 40), vectorize(1, 'sedmnact', 40), vectorize(6, 'vypravel', 40)]);
        print(testVector)
        
        scaler = MinMaxScaler()
        scaledTrain = scaler.fit_transform(trainVectors[3])
        scaledTest = scaler.transform(testVector)
        
        mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(150), max_iter=300)
        mlp.fit(scaledTrain, targets[3])
        
        print(mlp.predict_proba(scaledTest))
        print(mlp.predict(scaledTest))
        '''

        # TODO: Train a model on the given dataset and store it in `model`.
        model = models
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        #test = Dataset(args.predict)
        test = Dataset()

        with lzma.open(args.model_path, "rb") as model_file:
            models = pickle.load(model_file)

        predictions = ''
        ignored = [' ', '\n', '-', ':', ',', '.', '?', '!', '"']

        mapp = {
        'a' : ['a', 'á'],
        'c' : ['c', 'č'],
        'd' : ['d', 'ď'],
        'e' : ['e', 'é', 'ě'],
        'i' : ['i', 'í'],
        'n' : ['n', 'ň'],
        'o' : ['o', 'ó'],
        'r' : ['r', 'ř'],
        's' : ['s', 'š'],
        'u' : ['u', 'ú', 'ů'],
        'y' : ['y', 'ý'],
        'z' : ['z', 'ž'],
        }

        weights = {
        'a' : [1.283, 3.531],
        'c' : [1.669, 1.7131],
        'd' : [1.02, 10.526],
        'e' : [1.354, 8.964, 2.822],
        'i' : [1.634, 2.277],
        'n' : [1.008, 30.933],
        'o' : [1.002, 100.606],
        'r' : [1.4706, 2.325],
        's' : [1.205, 4.873],
        'u' : [1.141, 55.314, 4.505],
        'y' : [1.377, 4.654],
        'z' : [1.692, 2.446],
        }

        i = 0
        while i < len(test.data):
            if test.data[i] in ignored:
                predictions += test.data[i]
                i += 1
            else:
                startIndex = i
                endIndex = i+1
                while test.data[endIndex] not in ignored:
                    endIndex += 1

                word = test.data[startIndex:endIndex]

                wordList = list(word)
                for j in range(len(wordList)):
                    c = wordList[j]
                    if c in mapp.keys():
                        modelIndex = list(mapp.keys()).index(c)
                        vector = np.array([vectorize(j, word, 40)])
                        probas = models[modelIndex].predict_proba(vector)
                        probas = probas * weights[c]
                        newC = mapp[c][np.argmax(probas)]
                        #print(word, c, newC, probas)
                        wordList[j] = newC

                predictions += ''.join(wordList)

                i = endIndex

        f = open("pred.txt", "w", encoding='utf8')
        f.write(predictions)
        f.close()

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

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
parser.add_argument("--predict", default='fiction-test.txt', type=str, help="Run prediction on given data")
#parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

def makeVectors(text, diaText):
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
    text = text.replace('  ', ' ')
    '''
    text = text.replace(':', ' ')
    text = text.replace('"', ' ')
    text = text.replace(':', ' ')
    text = text.replace('?', ' ')
    text = text.replace(',', ' ')
    text = text.replace('-', ' ')
    text = text.replace('  ', ' ')
    '''
    diaText = diaText.lower()
    diaText = diaText.replace('\n', ' ');
    diaText = diaText.replace('  ', ' ')
    '''
    diaText = diaText.replace(':', ' ')
    diaText = diaText.replace('"', ' ')
    diaText = diaText.replace(':', ' ')
    diaText = diaText.replace('?', ' ')
    diaText = diaText.replace(',', ' ')
    diaText = diaText.replace('-', ' ')
    diaText = diaText.replace('  ', ' ')
    '''
    
    trainVectors = []
    targets = []
    for i in range(2):
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

    vectorLen = maxLen*2+1
    #print('vec len = ' + str(vectorLen))

    for i in range(len(words)):
        trainVectors[0].append(vectorize(words[i], vectorLen))
        trainVectors[1].append(vectorize(words[i-1] + ' ' + words[i], vectorLen))

        targets[0].append(diaWords[i])
        targets[1].append(diaWords[i])

    for i in range(2):
        trainVectors[i] = np.array(trainVectors[i])
        targets[i] = np.array(targets[i])

    trainVectors = np.array(trainVectors)
    targets = np.array(targets)

    return trainVectors, targets

def vectorize(word, vectorLen):
    #centerIndex = vectorLen // 2
    #centeredWord = list(' ' * vectorLen)
    #centeredWord[centerIndex-centerCharIndex:centerIndex-centerCharIndex+len(word)] = list(word)
    padding = vectorLen - len(word)
    word += ' '*padding
    ll = list(map(lambda x: ord(x), word))


    return np.array(ll)

def mymetric(x, y):
    return np.count_nonzero(x-y)


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_data, train_target = makeVectors(train.data, train.target)

        models = []
        c1 = KNeighborsClassifier(n_neighbors=1, p=2, n_jobs=-1)
        c2 = KNeighborsClassifier(n_neighbors=1, p=2, n_jobs=-1)

        c1.fit(train_data[0], train_target[0])
        c2.fit(train_data[1], train_target[1])

        models.append(c1)
        models.append(c2)
        '''
        model = Pipeline(steps = [
            #('trans', OneHotEncoder()),
            ('mlp', KNeighborsClassifier(n_neighbors=1, p=2, n_jobs=-1))
        ])
        model.fit(train_data, train_target)

        print(neigh.predict(np.array([vectorize('reci', 20)])))
        print(neigh.predict_proba(np.array([vectorize('reci', 20)])))


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
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            models = pickle.load(model_file)

        #ignored = [' ', '\n', '-', ':', ',', '.', '?', '!', '"']
        m1 = models[0]
        m2 = models[1]

        ignored = [' ', '\n']
        i = 0
        predictions = ''

        defaultWords = []
        words = []
        wordsLong = []
        wasCaps = []
        wasUpper = []
        testLower = test.data.lower()
        #testLower = testLower.replace('\n', ' ');
        #testLower = testLower.replace('  ', ' ')
        prevWord = ''
        while i < len(testLower):
            if testLower[i] in ignored:
                predictions += testLower[i]
                i += 1
            else:
                startIndex = i
                endIndex = i+1
                while testLower[endIndex] not in ignored:
                    endIndex += 1

                if endIndex == startIndex+1:
                    i = endIndex
                    predictions += test.data[startIndex:endIndex]
                    continue

                word = testLower[startIndex:endIndex]
                defaultWords.append(word)
                words.append(vectorize(word, 41))
                if (prevWord != ''):
                    wordsLong.append(vectorize(prevWord + ' ' + word, 41))
                else:
                    wordsLong.append(vectorize(word, 41))

                capsC = test.data[startIndex]

                if capsC == word[0]:
                    wasCaps.append(0)
                    wasUpper.append(0)
                else:
                    wasCaps.append(1)
                    if len(word) > 1 and startIndex+1< len(test.data) and test.data[startIndex+1] != word[1]:
                        wasUpper.append(1)
                    else:
                        wasUpper.append(0)
                i = endIndex

                prevWord = word

        predictions = ''
        dist1, newWords = m1.kneighbors(np.array(words), 1)
        dist2, newWordsLong = m2.kneighbors(np.array(wordsLong), 1)

        print(m1.kneighbors(np.array([vectorize('rady', 41)]), 2))
        print(m2.kneighbors(np.array([vectorize('si ' + 'rady', 41)]), 2))
        print(m1.classes_[m1._y[17339]])

        distt = DistanceMetric.get_metric('manhattan')
        
        print(distt.pairwise([vectorize('rady', 41)], [vectorize('rády', 41)]))
        wordIndex = 0
        i = 0
        while i < len(testLower):
            if testLower[i] in ignored:
                predictions += testLower[i]
                i += 1
            else:
                startIndex = i
                endIndex = i+1
                while testLower[endIndex] not in ignored:
                    endIndex += 1

                if endIndex == startIndex+1:
                    i = endIndex
                    predictions += test.data[startIndex:endIndex]
                    continue

                newWord = m1.classes_[m1._y[newWords[wordIndex]]][0]
                newWordLong = m2.classes_[m2._y[newWordsLong[wordIndex]]][0]

                dist = dist1[wordIndex][0]
                distd = dist2[wordIndex][0]

                if (newWord != newWordLong and distd == 0):
                    #diff = mymetric(vectorize(newWord, 20), vectorize(newWordLong, 20))
                    #if diff < 2:
                    #print(newWord, 'za', newWordLong)
                    newWord = newWordLong
                    dist = dist2[wordIndex][0]
                '''
                else: 
                    if dist > dist2[wordIndex][0]/1000:

                        dist = dist2[wordIndex][0]
                        newWord = newWordLong
                '''

                if defaultWords[wordIndex] == 'rady':
                    print(defaultWords[wordIndex-1], defaultWords[wordIndex])
                    print(newWord, newWordLong)
                    print(dist, distd)

                if dist > 0.1:
                    newWord = defaultWords[wordIndex]


                if wasCaps[wordIndex]:
                    newWord = newWord.capitalize()

                if wasUpper[wordIndex]:
                    newWord = newWord.upper()
                predictions += newWord
                wordIndex += 1
                i = endIndex
        
        i = 0
        count = 0
        prevWord = ''
        while i < len(predictions):
            if predictions[i] in ignored:
                i += 1
            else:
                startIndex = i
                endIndex = i+1
                while predictions[endIndex] not in ignored:
                    endIndex += 1

                word = predictions[startIndex:endIndex] 

                if word == 'že' and prevWord.lower() != 'ale' and predictions[startIndex-2] != ',' and predictions[startIndex-2].lower() != 'a' and predictions[startIndex-3] != ' ':
                    predictions = predictions[:startIndex] + 'ze' + predictions[endIndex:]

                if word == 'ze' and (prevWord.lower() == 'ale' or prevWord == ','):
                    predictions = predictions[:startIndex] + 'že' + predictions[endIndex:]

                if word == 'Že':
                    predictions = predictions[:startIndex] + 'Ze' + predictions[endIndex:]

                if word == 'ně' and predictions[startIndex-2] == ',':
                    predictions = predictions[:startIndex] + 'ne' + predictions[endIndex:]

                if word == 'mne' and (prevWord == 'ke' or prevWord == 'o'):
                    predictions = predictions[:startIndex] + 'mně' + predictions[endIndex:]
                if word == 'té' and (prevWord.lower() == 'prosím' or prevWord.lower() == 'jsme' or prevWord.lower() == 'aby' or prevWord.lower() == 'abych' or prevWord.lower() == 'co' or prevWord.lower() == 'kdo' or prevWord.lower() == 'který'):
                    predictions = predictions[:startIndex] + 'tě' + predictions[endIndex:]

                if word == 'Té' and (prevWord.lower() == 'prosím'):
                    predictions = predictions[:startIndex] + 'Tě' + predictions[endIndex:]
                i = endIndex

                prevWord = word
        
        f = open("pred.txt", "w", encoding='utf8')
        f.write(predictions)
        f.close()
        
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

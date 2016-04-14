from __future__ import division
from src.Perceptron import Perceptron
from CrossValidation import CrossValidation
import numpy as np

## Train a perceptron and update its weights
# @param trainData npdata field of Data object
def trainPerceptron(perceptron, trainData, learning_rate, num_epochs):
    perceptron.setLearningRate(learning_rate)
    for i in range(0, num_epochs):
        for instance in trainData:
            label = instance[-1]
            attributes = [float(x) for x in instance[:-1]]
            error = label - perceptron.getOutput([1] + attributes)
            perceptron.backPropUpdate([1] + attributes, error)


## given a trained perceptron, evaluate the acurracy rate on a test dataset
def evalAccuracy(perceptron, testData):
    results = []
    for instance in testData:
        label = instance[-1]
        attributes = instance[:-1].tolist()
        output = round(perceptron.getOutput([1] + attributes))
        if label == output:
            results.append(1)
        else:
            results.append(0)
    nplist = np.array(results)
    return nplist.mean()


## Calculate cross validation accuracy for a perceptron
def cvAccuracy(data, num_folds, learning_rate, num_epochs):
    cv = CrossValidation(data, num_folds)
    cv.stratify()
    accuracy = []
    for fold_i in range(1, num_folds + 1):
        perceptron = Perceptron()
        perceptron.setWeights([0.1 for i in range(0,len(data.names))])
        subdataTrain, subdataTest = cv.getSubdata(fold_i)
        trainPerceptron(perceptron, subdataTrain, learning_rate, num_epochs)
        accuracy.append(evalAccuracy(perceptron, subdataTest))
    nplist = np.array(accuracy)
    return nplist.mean()


## Calculate ROC coordinates on a dataset, with a trained perceptron
def getROC(data, learning_rate, num_folds, num_epochs):
    pass
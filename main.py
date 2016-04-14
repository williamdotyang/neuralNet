from src.Perceptron import Perceptron
from src.CrossValidation import CrossValidation
from src.evaluation import *
from tools.Data import Data
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 5: 
        sys.stderr('trainfile num_folds learning_rate num_epochs')
        sys.exit()

    trainfile, num_folds, learning_rate, num_epochs = sys.argv[1:]
    num_folds = int(num_folds)
    num_epochs = int(num_epochs)
    learning_rate = float(learning_rate)
    train = Data(trainfile)
    train.parse()

    cv = CrossValidation(train, num_folds)
    cv.stratify()
    results = {} # {dataIndex : [fold_of_instance, predicted_class, actual_class, confidence_of_prediction]
    for fold_i in range(1, num_folds + 1):
        subdataTrain, subdataTest = cv.getSubdata(fold_i)
        testDataIndicies = cv.subdataIndicies[fold_i]
        perceptron = Perceptron()
        perceptron.setWeights([0.1 for i in range(0,len(train.names))])
        trainPerceptron(perceptron, subdataTrain, learning_rate, num_epochs)
        for i in testDataIndicies:
            label = train.npdata[i][-1]
            if label == 1:
                label = train.variables["class"][-1]
            else:
                label = train.variables["class"][0]
            attributes = [float(x) for x in train.data[i][:-1]]
            confidence = perceptron.getOutput([1] + attributes)
            if int(round(confidence)) == 1:
                predicted = train.variables["class"][-1]
            else:
                predicted = train.variables["class"][0]
            results[i] = [fold_i, predicted, label, confidence]

    for index in sorted(results.keys()):
        print '\t'.join([str(x) for x in results[index]])
from src.Perceptron import Perceptron
from src.CrossValidation import CrossValidation
from src.evaluation import *
from tools.Data import Data
import numpy as np

if __name__ == "__main__":
    train = Data("sonar.arff")
    train.parse()

    print "--------- epochs 25, 50, 75 and 100 (learning rate = 0.1, number of folds = 10)"
    result = []
    for epoch in [25, 50, 75, 100]:
        result += [cvAccuracy(train, 10, 0.1, epoch)]
    print result
    print
    print "--------- number of folds as 5, 10, 15, 20 and 25 (learning_rate = 0.1, num_epochs = 50)"
    result = []
    for num_fold in [5, 10, 15, 20, 25]:
        result += [cvAccuracy(train, num_fold, 0.1, 50)]
    print result
    

    # PART B (3)
    num_folds = 10
    num_epochs = 50
    learning_rate = 0.1

    cv = CrossValidation(train, num_folds)
    cv.stratify()
    results = {} # {dataIndex : [actual_class, confidence_of_prediction]
    for fold_i in range(1, num_folds + 1):
        subdataTrain, subdataTest = cv.getSubdata(fold_i)
        testDataIndicies = cv.subdataIndicies[fold_i]
        perceptron = Perceptron()
        perceptron.setWeights([0.1 for i in range(0,len(train.names))])
        trainPerceptron(perceptron, subdataTrain, learning_rate, num_epochs)
        for i in testDataIndicies:
            label = train.npdata[i][-1]
            attributes = [float(x) for x in train.data[i][:-1]]
            confidence = perceptron.getOutput([1] + attributes)
            results[i] = [label, confidence]

    f1 = open("targets", "w+")
    f2 = open("outputs", "w+")
    for key in results.keys():
        target = results[key][0]
        output = results[key][1]
        f1.write(str(int(target)) + '\n')
        f2.write(str(output) + '\n')
    f1.close()
    f2.close()
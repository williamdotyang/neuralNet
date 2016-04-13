from __future__ import division
import math
import calc
import numpy as np

class Perceptron(object):
    """
    A class representation of a neural perceptron.
    """
    def __init__(self, name):
        self.name = name

    ## Set weights for the inputs and offset
    # weights = [weight for offset, weight for input1, ...]
    def setWeights(self, weights, verbose = False):
        self.weights = np.array(weights)
        if verbose:
            print self.name, "weights set", self.weights.tolist()

    ## Set learning rate
    def setLearningRate(self, learningRate, verbose = False):
        self.learningRate = learningRate
        if verbose:
            print self.name, "learningRate set", self.learningRate

    ## Calculated the output, using sigmoid function.
    # inputs = [1, input1, input2, ...]
    def getOutput(self, inputs, verbose = False):
        inputs = np.array(inputs)
        net = sum(self.weights * inputs)
        out = calc.sigmoid(net)
        if verbose:
            print self.name, "inputs:", inputs.tolist(), "output:", out
        return out

    ## Calculate the delta term for backprop updating.
    # errorContribution = (y - o), if this is an output unit,
    #            = sum(error_k * weight_k), if this is a hidden unit
    def getDelta(self, inputs, errorContribution):
        inputs = np.array(inputs)
        errorContribution = np.array(errorContribution)
        output = self.getOutput(inputs)
        return calc.sigmoid_der(output) * errorContribution

    ## Update the weights
    def backPropUpdate(self, inputs, errorContribution, verbose = False):
        delta = self.getDelta(inputs, errorContribution)
        delta_weights = self.learningRate * delta * inputs
        if verbose:
            print self.name, "old weights:", self.weights.tolist()
            print self.name, "delta:", delta
            print self.name, "weights change:", delta_weights
        self.weights = self.weights + delta_weights
        if verbose:
            print self.name, "new weights:", self.weights.tolist()


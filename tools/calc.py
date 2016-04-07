from __future__ import division
import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)
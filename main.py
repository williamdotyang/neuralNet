from src.Perceptron import Perceptron
from tools.Data import Data
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 5: 
        sys.stderr('trainfile num_folds learning_rate num_epochs')
        sys.exit()

    trainfile num_folds learning_rate num_epochs = sys.argv[1:]
    train = Data(trainfile)
    train.parse()
    
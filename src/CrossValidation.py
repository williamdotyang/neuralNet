from tools.Data import Data
import random 

class CrossValidation(object):
    """A calss representation of cross validation datasets."""

    ## subdata = {1:[instances for fold 1], 2:[instances for fold 2], ...}
    def __init__(self, train, num_folds):
        self.train = train
        self.num_folds = num_folds
        self.subdata = {}

    ## Make stratified cross validation datasets by setting up the subdata.
    def stratify(self, seed=1):
        group0 = []
        group1 = []
        label0 = self.train.variables["class"][0]
        for i in range(0, len(self.train.npdata)):
            label = self.train.data[i]
            if label == label0:
                group0.append(self.train.npdata[i])
            else:
                group1.append(self.train.npdata[i])

        for i in range(0, self.num_folds):
            self.subdata[i + 1] = []

        random.seed(seed)
        i = 0
        while group0 != []:
            randIndex = random.randint(0, len(group0) - 1)
            self.subdata[i].append(group0[randIndex])
            i = (i + 1) % self.num_folds
            del(group0[randIndex])

        i = 0
        while group1 != []:
            randIndex = random.randint(0, len(group1) - 1)
            self.subdata[i].append(group1[randIndex])
            i = (i + 1) % self.num_folds
            del(group1[randIndex])


from tools.Data import Data
import random 

class CrossValidation(object):
    """A calss representation of cross validation datasets."""

    ## subdata = {1:[instances for fold 1], 2:[instances for fold 2], ...}
    def __init__(self, train, num_folds):
        self.train = train
        self.num_folds = num_folds
        self.subdataIndicies = {} # {fold_i: [indicies in train.data]}
        self.index2fold = {} # {data_index:fold_i}

    ## Make stratified cross validation datasets by setting up the subdataIndicies.
    def stratify(self, seed=1):
        group0 = [] # [indicies of label0]
        group1 = [] # [indicies of label1]
        label0 = self.train.variables["class"][0]
        for i in range(0, len(self.train.data)):
            label = self.train.data[i][-1]
            if label == label0:
                group0.append(i)
            else:
                group1.append(i)

        for i in range(0, self.num_folds):
            self.subdataIndicies[i+1] = []

        random.seed(seed)
        i = 0
        while group0 != []:
            randIndex = random.randint(0, len(group0) - 1)
            indexInMatrix = group0[randIndex]
            self.subdataIndicies[i+1].append(indexInMatrix)
            self.index2fold[indexInMatrix] = i
            i = (i + 1) % self.num_folds
            del(group0[randIndex])

        i = 0
        while group1 != []:
            randIndex = random.randint(0, len(group1) - 1)
            indexInMatrix = group1[randIndex]
            self.subdataIndicies[i+1].append(indexInMatrix)
            self.index2fold[indexInMatrix] = i
            i = (i + 1) % self.num_folds
            del(group1[randIndex])

        #for i in range(0, self.num_folds):
        #    random.shuffle(self.subdataIndicies[i+1])

    ## based on subdataIndicies, create subdata matrix: train and test
    def getSubdata(self, fold):
        indicies = set(self.subdataIndicies[fold])
        subdataTest = []
        subdataTrain = []
        for i in range(0, len(self.train.data)):
            if i in indicies:
                subdataTest.append(self.train.data[i])
            else:
                subdataTrain.append(self.train.data[i])
        random.shuffle(subdataTrain)
        random.shuffle(subdataTest)
        return subdataTrain, subdataTest
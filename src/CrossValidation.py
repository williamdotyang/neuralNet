import random 

class CrossValidation(object):
    """A calss representation of cross validation datasets."""

    ## subdataIndicies = {1:[indicies for fold 1], 2:[indicies for fold 2], ...}
    def __init__(self, train, num_folds):
        self.train = train
        self.num_folds = num_folds
        self.subdataIndicies = None

    ## Make stratified cross validation datasets by setting up the subdataIndicies.
    def stratify(self):
        pass


import re
import numpy as np

class Data:
    """ A class representation of dataset, either training or testing.
        All the features are continuous, and class variable is binary.
        Provides some basic counting functions of instances.
    """

    ##
    # @param filename  A string of input file name.
    def __init__(self, filename):
        self.filename = filename
        # A list of variable names representing the order [attr1, attr2, ...]
        self.names = []
        # A dictionary of variables {V1:[v11,v12,...], V2:[v21, v22,...], ...}
        self.variables = {} 
        # A matrix representing the instances [[..],[..],...]
        self.data = []

    ##
    # Reads a file and set some fields with data structures for easier access.
    def parse(self):
        for line in open(self.filename):
            line = line.rstrip('\n\r')
            if line[0] == '%':
                pass
            elif line[0] == '@':
                tokens = line.split(None, 2)
                if tokens[0] == "@attribute":
                    attribute = tokens[1].strip("'").lower()
                    self.names.append(attribute)
                    self.variables[attribute] = re.split(', *', tokens[2].strip("{ }"))
            else: # data lines
                self.data.append(line.split(','))

        # convert all the attributes and class label into numeric
        self.npdata = []
        label1 = self.variables['class'][1] # class label that to be classified as 1
        for line in self.data:
            attributes = line[0:-1]
            label = line[-1]
            if label == label1:
                label = 1
            else:
                label = 0
            self.npdata.append([float(x) for x in attributes] + [label])
        self.npdata = np.array(self.npdata)


    ##
    # Get the number of variables in this data set.
    def getNumNames(self):
        return len(self.names)

    ##
    # Get the number of instances in this data set.
    def getNumInstances(self):
        return len(self.data)

    ##
    # Get the column index(es) of given variables.
    # @param variables  A list of names of variables whose colum indicies are asked
    # @return A list of int
    def getColIndex(self, variables):
        indicies = []
        for var in variables:
            i = 0
            for name in self.names:
                if name == var:
                    indicies.append(i)
                    break
                else:
                    i += 1
        return indicies

    ##
    # Count the number of instances where given set of variables having a 
    # given set of values.
    # @param X  A list of names of selected variables.
    # @param v  A list of values for each of the variables.
    # @return An int, the count of instances in data where X=v.)
    def countInstances(self, X, v):
        cols = self.getColIndex(X)
        count = 0
        for line in self.data:
            x = [line[i] for i in cols]
            if x == v:
                count += 1
        return count

    ## overrided __str__ method
    def __str__(self):
        print self.filename
        print self.names
        print self.variables
        print self.data[1]


if __name__ == '__main__':
    # test code
    train = Data('sonar.arff')
    train.parse()
    print train.names
    print train.variables
    print train.data[1]
    print train.npdata[1]
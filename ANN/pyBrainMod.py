from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator
from numpy import *
import csv

def placeDataIntoArray(fileName):
    '''Add comma-delimited file content to a data array'''
    try:
        with open(fileName, mode='rbU') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
            dataArray = array([row for row in datareader], dtype=float64, order='C')

        if (min(dataArray.shape) == 1): # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray;
    except:
        print "error placing data from file '{0}' into array".format(str(fileName));

def getAllOfTheData(trainDataPath, train_pIC50Path, validationDataPath, validation_pIC50Path, testDataPath, test_pIC50Path):
    '''Gets training, test and cross-validation data set(s)'''
    try:
        TrainX    = placeDataIntoArray(trainDataPath);
        TrainY    = placeDataIntoArray(train_pIC50Path);
        ValidateX = placeDataIntoArray(validationDataPath);
        ValidateY = placeDataIntoArray(validation_pIC50Path);
        TestX     = placeDataIntoArray(testDataPath);
        TestY     = placeDataIntoArray(test_pIC50Path);
        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY;
    except:
        print "error getting all of data";

def rescaleTheData(TrainX, TrainY, TestX, TestY):
    try:
        # 1 degree of freedom means (ddof) N-1 unbiased estimation
        TrainXVar  = TrainX.var(axis = 0, ddof=1)
        TrainXMean = TrainX.mean(axis = 0)
        TrainYVar  = TrainY.var(axis = 0, ddof=1)
        TrainYMean = TrainY.mean(axis = 0);

        for i in range(0, TrainX.shape[0]):
            TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    
        for i in range(0, TrainX.shape[0]):
            TrainY[i] = (TrainY[i] - TrainYMean)/sqrt(TrainYVar)

        for i in range(0, TestX.shape[0]):
            TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

        for i in range(0, TestY.shape[0]):
            TestY[i] = (TestY[i] - TrainYMean)/sqrt(TrainYVar)

        return TrainX, TrainY, TestX, TestY
    except:
        print "error rescaling input data";

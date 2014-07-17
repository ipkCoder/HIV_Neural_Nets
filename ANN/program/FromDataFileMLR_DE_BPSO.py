import time                 # provides timing for benchmarks
from numpy  import *        # provides complex math and array functions
#from sklearn import svm	    # provides Support Vector Regression
import csv
import math
import sys
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "2091/data.txt"
abs_file_path = os.path.join(script_dir, rel_path)

#------------------------------------------------------------------------------
def getTwoDecPoint(x):
    return float("%.2f"%x);

#------------------------------------------------------------------------------
def placeDataIntoArray(fileName):
    try:
        with open(fileName, mode='rbU') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
            dataArray = array([row for row in datareader], dtype=float64, order='C')

        if (min(dataArray.shape) == 1): # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray;
        # dataArray = genfromtxt(filename, delimiter=',')
        # print dataArrayann
        # return dataArray
    except:
        print "error placing data into array for {}.".format(fileName)
        
#------------------------------------------------------------------------------
def getAllOfTheData():
    try:
        TrainX    = placeDataIntoArray(os.path.join(os.getcwd(), 'new_data/train_x.csv'))
        TrainY    = placeDataIntoArray(os.path.join(os.getcwd(), 'new_data/train_y.csv'))
        ValidateX = placeDataIntoArray(os.path.join(os.getcwd(), 'new_data/validation_x.csv'))
        ValidateY = placeDataIntoArray(os.path.join(os.getcwd(), 'new_data/validation_y.csv'))
        TestX     = placeDataIntoArray(os.path.join(os.getcwd(), 'new_data/test_x.csv'))
        TestY     = placeDataIntoArray(os.path.join(os.getcwd(), 'new_data/test_y.csv'))
        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY
    except:
        print "error getting all of data"

#------------------------------------------------------------------------------

def rescaleTheData(TrainX, ValidateX, TestX):
    try:
        # 1 degree of freedom means (ddof) N-1 unbiased estimation
        TrainXVar = TrainX.var(axis = 0, ddof=1)
        TrainXMean = TrainX.mean(axis = 0)

        for i in range(0, TrainX.shape[0]):
            TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
        for i in range(0, ValidateX.shape[0]):
            ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/sqrt(TrainXVar)
        for i in range(0, TestX.shape[0]):
            TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

        return TrainX, ValidateX, TestX;
    except:
        print "error rescaling the data"

#------------------------------------------------------------------------------

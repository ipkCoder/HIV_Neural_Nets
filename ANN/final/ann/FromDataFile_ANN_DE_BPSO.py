from numpy  import *        # provides complex math and array functions
import csv
import math
import sys
import os

'''
Purpose: load data into array
'''
def placeDataIntoArray(fileName):
    try:
        with open(fileName, mode='rbU') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
            dataArray = array([row for row in datareader], dtype=float64, order='C')

        if (min(dataArray.shape) == 1): # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray;
    except:
        print "error placing data into array for {}.".format(fileName)
        
'''
Purpose: create train, validate, test sets
'''
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

'''
Purpose: normalize the feature data
'''
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


#-------------------------------------------------------------------------------
# Name: cs-qsar-ann-modeling-research
# Purpose: Common functions library for use in the context of QSAR ANN 
# model training.
#
# Created: 20141701
# Copyright: (c) CSUSM 2014
# Licence: MIT
#-------------------------------------------------------------------------------
from numpy import *
import csv
from collections import namedtuple
import os
import datetime

# Since we can't declare constants in Python, so we use a namedtuple
# to define immutable index value(s) for log entry types.  A namedtuple
# throws an error if a tuple value is updated after initialization.
Constants      = namedtuple('constants', ['info','warning','error'])
logtype        = Constants(0,1,2);
reverseLookup  = {0:"info",1:"warning",2:"error"};

def log(lineEntry = "", category = logtype.info):
    '''Writes non-empty lines to a log file in the relative
        directory of the ANN module. Format tracelog_{YYYY-MM-DD}.log'''
    try:
        # don't log empty lines
        if not lineEntry:
            return 0;

        localCategory = category;
        if reverseLookup.has_key(localCategory) is False:
                # Default to "info" log entry type
                localCategory = log.info;
        logName          = "tracelog_{0}.log".format(datetime.datetime.now().strftime("%Y-%m-%d"));
        logFile          = os.path.join(os.getcwd(), logName);
        entryDate        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f");
        # appends to existing log file, else writes a new log file
        with open(logFile,"a") as log:
            log.writelines("".join([entryDate, "\t", reverseLookup[localCategory], "\t", lineEntry, "\n"]));
        return 0;
    except:
        print "error writing log entry: {0} to log file {1}".format(lineEntry,logFile);

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

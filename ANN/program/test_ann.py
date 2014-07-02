from ANN import ANN
import csv
import time
from numpy import *
from pybrain.tools.validation import Validator

def r2(y, yHat):
    """Coefficient of determination"""
    numer = ((y - yHat)**2).sum()       # Residual Sum of Squares
    denom = ((y - y.mean())**2).sum()
    r2 = 1 - numer/denom
    return r2

def rmse(X, Y):
	return (sum((X-Y)**2)/len(X))**.5

def ccc(y, yHat):
    """Concordance Correlation Coefficient"""
    n = y.shape[0]
    numer = 2*(((y - y.mean())*(yHat - yHat.mean())).sum())
    denom = ((y - y.mean())**2).sum() + ((yHat - yHat.mean())**2).sum() + n*((y.mean() - yHat.mean())**2)
    ccc = numer/denom
    return ccc

def placeDataIntoArray(fileName):
    with open(fileName, mode='rbU') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        dataArray = array([row for row in datareader], dtype=float64, order='C')

    if (min(dataArray.shape) == 1): # flatten arrays of one row or column
        return dataArray.flatten(order='C')
    else:
        return dataArray

def getAllOfTheData():
    TrainX = placeDataIntoArray('Train-Data.csv')
    TrainY = placeDataIntoArray('Train-pIC50.csv')
    ValidateX = placeDataIntoArray('Validation-Data.csv')
    ValidateY = placeDataIntoArray('Validation-pIC50.csv')
    TestX = placeDataIntoArray('Test-Data.csv')
    TestY = placeDataIntoArray('Test-pIC50.csv')
    return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY

def rescaleTheData(TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):

    # 1 degree of freedom means (ddof) N-1 unbiased estimation
    TrainXVar = TrainX.var(axis = 0, ddof=1)
    TrainXMean = TrainX.mean(axis = 0)
    TrainYVar = TrainY.var(axis = 0, ddof=1)
    TrainYMean = TrainY.mean(axis = 0)
	
    # for i in range(0, TrainX.shape[0]):
    #     TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    
    # for i in range(0, TrainX.shape[0]):
    #     TrainY[i] = (TrainY[i] - TrainYMean)/sqrt(TrainYVar)

    # for i in range(0, TestX.shape[0]):
    #     TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

    # for i in range(0, TestY.shape[0]):
    #     TestY[i] = (TestY[i] - TrainYMean)/sqrt(TrainYVar)

    # return TrainX, TrainY, TestX, TestY

    for i in range(TrainX.shape[0]):
        TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    for i in range(ValidateX.shape[0]):
        ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/sqrt(TrainXVar)
    for i in range(TestX.shape[0]):
        TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

    return TrainX, ValidateX, TestX

TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = getAllOfTheData()

# combine test and training sets
# Backprop trainer will split it up
# TrainValX = append(TrainX, ValidateX, axis=0)
# TrainValY = append(TrainY, ValidateY, axis=0)

# rescale data
TrainX, ValidateX, TestX = rescaleTheData(TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
# TrainValX, TrainValY, TestX, TestY = rescaleTheData(TrainValX, TrainValY, TestX, TestY)

ann = ANN()

ann.create_network(TrainX.shape[1], 20, 1)

ann.train(TrainX, TrainY, ValidateX, ValidateY)
# ann.train(TrainValX, TrainValY)

predictions = ann.predict(TestX)

for i in range(TestX.shape[0]):
    print predictions[i], TestY[i]
print "MSE: {}".format(rmse(predictions, TestY))
print "Corr: {}".format(ccc(TestY, predictions))
print "R2: {}".format(r2(TestY, predictions))

# test_outputs = zeros(TrainValX.shape[0])
# for j in range(TrainValX.shape[0]):
# 	test_outputs[j] = ann.predict(TrainValX[j])
# 	print test_outputs[j], TrainValY[j]
# print "MSE: {}".format(rmse(test_outputs, TrainValY))
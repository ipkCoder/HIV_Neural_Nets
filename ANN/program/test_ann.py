from ANN import ANN
import csv
import time
from numpy import *
from pybrain.tools.validation import Validator
import FromDataFileMLR_DE_BPSO as fdf

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

TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = fdf.getAllOfTheData()

# combine test and training sets
# Backprop trainer will split it up
# TrainValX = append(TrainX, ValidateX, axis=0)
# TrainValY = append(TrainY, ValidateY, axis=0)

# rescale data
TrainX, ValidateX, TestX = fdf.rescaleTheData(TrainX, ValidateX, TestX)
# TrainValX, TrainValY, TestX, TestY = rescaleTheData(TrainValX, TrainValY, TestX, TestY)

ann = ANN()

ann.create_network(TrainX.shape[1], 20, 1)

train_errors, val_errors = ann.train(TrainX, TrainY, ValidateX, ValidateY)
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

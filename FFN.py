from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator
from numpy import *
import csv
import time

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

def rescaleTheData(TrainX, TrainY, TestX, TestY):

    # 1 degree of freedom means (ddof) N-1 unbiased estimation
    TrainXVar = TrainX.var(axis = 0, ddof=1)
    TrainXMean = TrainX.mean(axis = 0)
    TrainYVar = TrainY.var(axis = 0, ddof=1)
    TrainYMean = TrainY.mean(axis = 0)
	
    for i in range(0, TrainX.shape[0]):
        TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    
    for i in range(0, TrainX.shape[0]):
        TrainY[i] = (TrainY[i] - TrainYMean)/sqrt(TrainYVar)

    for i in range(0, TestX.shape[0]):
        TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

    for i in range(0, TestY.shape[0]):
        TestY[i] = (TestY[i] - TrainYMean)/sqrt(TrainYVar)

    return TrainX, TrainY, TestX, TestY

TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = getAllOfTheData()

# combine test and training sets
# Backprop trainer will split it up
TrainValX = append(TrainX, ValidateX, axis=0)
TrainValY = append(TrainY, ValidateY, axis=0)

# rescale data
TrainValX, TrainValY, TestX, TestY = rescaleTheData(TrainValX, TrainValY, TestX, TestY)

# network object
ffn = FeedForwardNetwork()

# create layers
inLayer = LinearLayer(TrainValX.shape[1], name="input")
hiddenLayer = SigmoidLayer(20, name="hidden1")
hiddenLayer2 = SigmoidLayer(20, name="hidden2")
outLayer = LinearLayer(1, name="output")

# add layers to feed forward network
ffn.addInputModule(inLayer)
ffn.addModule(hiddenLayer)
ffn.addModule(hiddenLayer2)
ffn.addOutputModule(outLayer)

# add bias unit to layers
ffn.addModule(BiasUnit(name='bias'))

# establish connections between layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_hidden = FullConnection(hiddenLayer, hiddenLayer2)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# add connections to network
ffn.addConnection(in_to_hidden)
ffn.addConnection(hidden_to_hidden)
ffn.addConnection(hidden_to_out)

# necessary, sort layers into correct/certain order
ffn.sortModules()

#ffn.convertToFastNetwork()

# print "Input layer weights: {}".format(in_to_hidden.params)
# print "Hidden layer weights: {}".format(hidden_to_out.params)
# print "All weights: {}".format(ffn.params)

# dataset object
ds = SupervisedDataSet(TrainValX.shape[1],1)

# add data to dataset object (ds)
for i in range(TrainValX.shape[0]):
	# print data[i]
	# print y[i]
	ds.addSample(TrainValX[i], TrainValY[i])

# Backprop trainer object
trainer = BackpropTrainer(ffn, ds)#, learningrate=.4, momentum=.2)#, verbose=True)

# learning rates to test
# note: best rates have been between .07 and .1, but seems to very inbetween
# most consistent between .07 and .085
alpha = array([0, .05, .07, .0775, .085, .1, .2])#, .15, .2, .25, .3, 3.5])
momentum = array([.05, .1, .15])

outcsv = open('results.csv', 'w')
writer = csv.writer(outcsv)

writer.writerow(['alpha', 'momentum', 'trainMSE', 'testMSE'])

info = zeros(5)

trials = 10
# do n tests of training combos, average results
for n in range(trials):

	info[0] = n
	# test learning rates
	for i in range(alpha.shape[0]):
		info[1] = alpha[i]

		# randomiz weights
		ffn.randomize()

		for k in range(momentum.shape[0]):

			info[2] = momentum[k]

			# Backprop trainer object
			trainer = BackpropTrainer(ffn, ds, learningrate=alpha[i], momentum=momentum[k])#, verbose=True)

			# for j in range(1000):
			# 	error = trainer.train()
			# 	if j%10 == 0:
			# 		print "{} error: {}".format(j, error)
			# 	# print "All weights: {}".format(ffn.params)
			# 	if error < .001:
			# 		break 

			# splits data into 75% training, 25% validation
			# train until convergence
			error = trainer.trainUntilConvergence(maxEpochs=2000, continueEpochs=10)# validationPortion=.X

			# print results
			print "alpha: {}, momentum: {}".format(alpha[i], momentum[k])

			train_outputs = zeros(TrainValX.shape[0])
			for j in range(TrainValX.shape[0]):
				train_outputs[j] = ffn.activate(TrainValX[j])
				# print train_outputs[j], TrainValY[j]
			mse = Validator.MSE(train_outputs, TrainValY)
			info[3] = mse
			#print "Train MSE: {}".format(mse)

			test_outputs = zeros(TestX.shape[0])
			for j in range(TestX.shape[0]):
				test_outputs[j] = ffn.activate(TestX[j])
				# print test_outputs[j], TestY[j]
			mse = Validator.MSE(test_outputs, TestY)
			info[4] = mse
			#print "Test MSE: {}".format(mse)
			writer.writerow(info)




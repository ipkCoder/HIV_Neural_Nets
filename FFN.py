from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from numpy import *
import csv

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

TrainValX = append(TrainX, ValidateX, axis=0)
TrainValY = append(TrainY, ValidateY, axis=0)

TrainValX, TrainValY, TestX, TestY = rescaleTheData(TrainValX, TrainValY, TestX, TestY)

ffn = FeedForwardNetwork()

# create layers
inLayer = LinearLayer(TrainValX.shape[1], name="input")
hiddenLayer = SigmoidLayer(20, name="hidden1")
# hiddenLayer2 = SigmoidLayer(20, name="hidden2")
outLayer = LinearLayer(1, name="output")

# add layers to feed forward network
ffn.addInputModule(inLayer)
ffn.addModule(hiddenLayer)
# ffn.addModule(hiddenLayer2)
ffn.addOutputModule(outLayer)

ffn.addModule(BiasUnit(name='bias'))
# establish connections between layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
# hidden_to_hidden = FullConnection(hiddenLayer, hiddenLayer2)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# add connections to network
ffn.addConnection(in_to_hidden)
# ffn.addConnection(hidden_to_hidden)
ffn.addConnection(hidden_to_out)

# necessary, sort layers into correct/certain order
ffn.sortModules()

ffn.convertToFastNetwork()

ffn.randomize()

# print ffn.activate([1, 2])


print TrainX
# print "Input layer weights: {}".format(in_to_hidden.params)
# print "Hidden layer weights: {}".format(hidden_to_out.params)
# print "All weights: {}".format(ffn.params)

ds = SupervisedDataSet(TrainValX.shape[1],1)

# data = np.random.randn(1000, 2)
# y = np.random.randn(1000, 1)

# add data to dataset object (ds)
for i in range(TrainValX.shape[0]):
	# print data[i]
	# print y[i]
	ds.addSample(TrainValX[i], TrainValY[i])

# trainer = BackpropTrainer(ffn, ds, learningrate=.4, momentum=.2)#, verbose=True)
trainer = BackpropTrainer(ffn, ds)#, learningrate=.4, momentum=.2)#, verbose=True)

alpha = array([.05, .1, .15, .2, .25, .3, 3.5])

for i in range(alpha.shape[0]):
	trainer = BackpropTrainer(ffn, ds, learningrate=alpha[i]) 

	# for j in range(1000):
	# 	error = trainer.train()
	# 	if j%10 == 0:
	# 		print "{} error: {}".format(j, error)
	# 	# print "All weights: {}".format(ffn.params)
	# 	if error < .001:
	# 		break 

	error = trainer.trainUntilConvergence(maxEpochs=2000, continueEpochs=10)

	print "alpha: {}\n".format(alpha[i])
	for j in range(TrainValX.shape[0]):
		print ffn.activate(TrainValX[j]), TrainValY[j]

	print "\n"
	for j in range(TestX.shape[0]):
		print ffn.activate(TestX[j]), TestY[j]
# print ffn.activate(data[0])



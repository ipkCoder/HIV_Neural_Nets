from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np

# def placeDataIntoArray(fileName):
#     with open(fileName, mode='rbU') as csvfile:
#         datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
#         dataArray = array([row for row in datareader], dtype=float64, order='C')

#     if (min(dataArray.shape) == 1): # flatten arrays of one row or column
#         return dataArray.flatten(order='C')
#     else:
#         return dataArray

# def getAllOfTheData():
#     TrainX = placeDataIntoArray('Train-Data.csv')
#     TrainY = placeDataIntoArray('Train-pIC50.csv')
#     ValidateX = placeDataIntoArray('Validation-Data.csv')
#     ValidateY = placeDataIntoArray('Validation-pIC50.csv')
#     TestX = placeDataIntoArray('Test-Data.csv')
#     TestY = placeDataIntoArray('Test-pIC50.csv')
#     return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY

# def rescaleTheData(TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):

#     # 1 degree of freedom means (ddof) N-1 unbiased estimation
#     TrainXVar = TrainX.var(axis = 0, ddof=1)
#     TrainXMean = TrainX.mean(axis = 0)

#     for i in range(0, TrainX.shape[0]):
#         TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    
#     for i in range(0, TrainX.shape[0]):
#         TrainY[i] = (TrainY[i] - TrainYMean)/sqrt(TrainYVar)
    
#     for i in range(0, ValidateX.shape[0]):
#         ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/sqrt(TrainXVar)
    
#     for i in range(0, TestX.shape[0]):
#         TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

#     return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY

# TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFilePLSR_DE_BPSO.getAllOfTheData()
# TrainX, ValidateX, TestX = FromDataFilePLSR_DE_BPSO.rescaleTheData(TrainX, ValidateX, TestX)


ffn = FeedForwardNetwork()

# create layers
inLayer = LinearLayer(2, name="input")
hiddenLayer = SigmoidLayer(10, name="hidden1")
hiddenLayer2 = SigmoidLayer(10, name="hidden2")
outLayer = LinearLayer(1, name="output")

# add layers to feed forward network
ffn.addInputModule(inLayer)
ffn.addModule(hiddenLayer)
ffn.addModule(hiddenLayer2)
ffn.addOutputModule(outLayer)

ffn.addModule(BiasUnit(name='bias'))
# establish connections between layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_hidden = FullConnection(hiddenLayer, hiddenLayer2)
hidden_to_out = FullConnection(hiddenLayer2, outLayer)

# add connections to network
ffn.addConnection(in_to_hidden)
ffn.addConnection(hidden_to_hidden)
ffn.addConnection(hidden_to_out)

# necessary, sort layers into correct/certain order
ffn.sortModules()

ffn.convertToFastNetwork()

ffn.randomize()

print ffn.activate([1, 2])

print "Input layer weights: {}".format(in_to_hidden.params)
print "Hidden layer weights: {}".format(hidden_to_out.params)
print "All weights: {}".format(ffn.params)

ds = SupervisedDataSet(2,1)

data = np.random.randn(1000, 2)
y = np.random.randn(1000, 1)

# add data to dataset object (ds)
for i in range(data.shape[0]):
	# print data[i]
	# print y[i]
	ds.addSample(data[i], y[i])

trainer = BackpropTrainer(ffn, ds, learningrate=.4, momentum=.2)#, verbose=True)

for i in range(10):
	error = trainer.train()
	if i%10 == 0:
		print "{} error: {}".format(i, error)
	# print "All weights: {}".format(ffn.params)
	if error < .45:
		break 

# error = trainer.trainUntilConvergence(maxEpochs=1000)

print "\n"
for i in range(data.shape[0]):
	print ffn.activate(data[i]), y[i]
# print ffn.activate(data[0])



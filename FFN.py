from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np

ffn = FeedForwardNetwork()

# create layers
inLayer = LinearLayer(2, name="input")
hiddenLayer = SigmoidLayer(3, name="hidden1")
hiddenLayer2 = SigmoidLayer(3, name="hidden2")
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

trainer = BackpropTrainer(ffn, ds, learningrate=.4, momentum=.4, verbose=True)

for i in range(1000):
	error = trainer.train()
	print "error: {}".format(error)
	# print "All weights: {}".format(ffn.params)
	if error < .45:
		break 

# error = trainer.trainUntilConvergence(maxEpochs=1000)

# print "\n"
# print ffn.activateOnDataset(ds)
# print ffn.activate(data[0])



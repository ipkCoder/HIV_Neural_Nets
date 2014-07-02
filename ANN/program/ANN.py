from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator
import numpy as np

class ANN:
	
	def create_network(this, nFeatures, hidden1Size, nClasses):# hidden2Size, )
		# create network object
		this.ffn = FeedForwardNetwork()
		
		# create layer objects
		inLayer = LinearLayer(nFeatures, name="input")
		hiddenLayer = SigmoidLayer(hidden1Size, name="hidden1")
		# hiddenLayer2 = SigmoidLayer(hidden2Size, name="hidden2")
		outLayer = LinearLayer(nClasses, name="output")
		
		# add layers to feed forward network
		this.ffn.addInputModule(inLayer)
		this.ffn.addModule(hiddenLayer)
		# this.ffn.addModule(hiddenLayer2)
		this.ffn.addOutputModule(outLayer)

		# add bias unit to layers
		this.ffn.addModule(BiasUnit(name='bias'))

		# establish connections between layers
		in_to_hidden = FullConnection(inLayer, hiddenLayer)
		# hidden_to_hidden = FullConnection(hiddenLayer, hiddenLayer2)
		hidden_to_out = FullConnection(hiddenLayer, outLayer)

		# add connections to network
		this.ffn.addConnection(in_to_hidden)
		# this.ffn.addConnection(hidden_to_hidden)
		this.ffn.addConnection(hidden_to_out)

		# necessary, sort layers into correct/certain order
		this.ffn.sortModules()
		# randomiz weights
		this.ffn.randomize()
		
		# dataset object
		this.train_ds = SupervisedDataSet(nFeatures, nClasses)
		this.validate_ds = SupervisedDataSet(nFeatures, nClasses)

	# train network
	def train(this, TrainX, TrainY, ValidateX, ValidateY):
		# should clear dataset

		# add data to dataset object (ds)
		for i in range(TrainX.shape[0]):
			this.train_ds.addSample(TrainX[i], TrainY[i])

		for i in range(ValidateX.shape[0]):
			this.validate_ds.addSample(ValidateX[i], ValidateY[i])

		# Backprop trainer object
		this.trainer = BackpropTrainer(this.ffn, learningrate=.0775, momentum=.05)
		
		this.trainer.trainUntilConvergence(trainingData=this.train_ds, validationData=this.validate_ds, maxEpochs=500, continueEpochs=10)

		return 'ANN'

	# predict depenent variable for dataset
	def predict(this, data):
		
		outputs = np.zeros(data.shape[0])
		for i in range(data.shape[0]):
			outputs[i] = this.ffn.activate(data[i])
		return outputs




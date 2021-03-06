from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator

import numpy as np
from qsarHelpers import *

class ANN:
    
    def __init__(self):
        self.name="ANN"
        
    def getParams(self):
        return self.in_to_hidden.params, self.hidden_to_out.params
    
    def create_network(self, nFeatures, hidden1Size=20, nClasses=1):
        # create network object
        self.ffn = FeedForwardNetwork()

        # create layer objects
        inLayer = LinearLayer(nFeatures, name="input")
        hiddenLayer = SigmoidLayer(hidden1Size, name="hidden1")
        #hiddenLayer2 = SigmoidLayer(hidden2Size, name="hidden2")
        outLayer = LinearLayer(nClasses, name="output")
            
        # add layers to feed forward network
        self.ffn.addInputModule(inLayer)
        self.ffn.addModule(hiddenLayer)
        #self.ffn.addModule(hiddenLayer2)
        self.ffn.addOutputModule(outLayer)

        # add bias unit to layers
        self.ffn.addModule(BiasUnit(name='bias'))

        # establish connections between layers
        self.in_to_hidden = FullConnection(inLayer, hiddenLayer)
        #hidden_to_hidden = FullConnection(hiddenLayer, hiddenLayer2)
        self.hidden_to_out = FullConnection(hiddenLayer, outLayer)

        # print "into hidden: {}".format(len(in_to_hidden.params))
        # print "into out: {}".format(len(hidden_to_out.params))
        
        # add connections to network
        self.ffn.addConnection(self.in_to_hidden)
        #self.ffn.addConnection(hidden_to_hidden)
        self.ffn.addConnection(self.hidden_to_out)

        # necessary, sort layers into correct/certain order
        self.ffn.sortModules()
        
        # dataset object
        self.train_ds = SupervisedDataSet(nFeatures, nClasses)
        self.validate_ds = SupervisedDataSet(nFeatures, nClasses)

    # train network
    def train(self, TrainX, TrainY, ValidateX, ValidateY):
        # clear old dataset
        self.train_ds.clear()
        self.validate_ds.clear()

        # add data to dataset object (ds)
        for i in range(TrainX.shape[0]):
            self.train_ds.addSample(TrainX[i], TrainY[i])

        for i in range(ValidateX.shape[0]):
            self.validate_ds.addSample(ValidateX[i], ValidateY[i])

        # randomiz weights
        self.ffn.randomize()

        # Backprop trainer object
        self.trainer = BackpropTrainer(self.ffn, learningrate=.0775, momentum=.1)
        try:
            with Timer() as t:
                self.train_errors, self.val_errors \
                    = self.trainer.trainUntilConvergence(trainingData=self.train_ds, \
                                                         validationData=self.validate_ds, \
                                                         maxEpochs=500, \
                                                         continueEpochs=10)

            return self.train_errors, self.val_errors
        except:
            print "Error occured while training model in ANN."
        
        #finally:
        #    print("ANN.py - Time to trainUntilConvergence: {:.03f} sec.".format(t.interval))

        #return 'ANN'

    # predict depenent variable for dataset
    def predict(self, data):
        # if only make prediction for one sample
        if(len(data.shape) == 1):
            return self.ffn.activate(data)
        else:
            outputs = np.zeros(data.shape[0])
            for i in range(data.shape[0]):
                outputs[i] = self.ffn.activate(data[i])
            return outputs



# ============ Test class =============
# if __name__ == '__main__':
#     import FromDataFile_ANN_DE_BPSO as fdf
#     TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = fdf.getAllOfTheData()
#     TrainX, ValidateX, TestX                           = fdf.rescaleTheData(TrainX, ValidateX, TestX)

#     ann = ANN()
#     ann.create_network(TrainX.shape[1], 20, 1)
#     train_errors, val_errors = ann.train(TrainX, TrainY, ValidateX, ValidateY)
#     predictions = ann.predict(TestX)

#     for i in range(TestX.shape[0]):
#         print predictions[i], TestY[i]
    

# ============== pycallgraph ===============
# from pycallgraph import PyCallGraph
# from pycallgraph import Config
# from pycallgraph import GlobbingFilter
# from pycallgraph.output import GraphvizOutput

# config = Config(max_depth=4)
# config.trace_filter = GlobbingFilter(
#         exclude=[
#         'pycallgraph.*',
#     ])
    
# graphviz = GraphvizOutput(output_file='ANN.png')

# with PyCallGraph(output=graphviz, config=config):
    
#     TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = fdf.getAllOfTheData()
#     TrainX, ValidateX, TestX                           = fdf.rescaleTheData(TrainX, ValidateX, TestX)

#     ann = ANN()
#     ann.create_network(TrainX.shape[1], 20, 1)
#     train_errors, val_errors = ann.train(TrainX, TrainY, ValidateX, ValidateY)
#     predictions = ann.predict(TestX)







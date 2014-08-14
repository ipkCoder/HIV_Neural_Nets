from ANN import ANN
import time
from numpy import *
import FromDataFileMLR_DE_BPSO as fdf

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput

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

config = Config()
config.trace_filter = GlobbingFilter(exclude=[
        'pycallgraph.*',
    ])
    
graphviz = GraphvizOutput(output_file='filter_exclude.png')
with PyCallGraph(output=graphviz, config=config):
    
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = fdf.getAllOfTheData()
    TrainX, ValidateX, TestX = fdf.rescaleTheData(TrainX, ValidateX, TestX)

    ann = ANN()

    ann.create_network(TrainX.shape[1], 20, 1)

    train_errors, val_errors = ann.train(TrainX, TrainY, ValidateX, ValidateY)

    predictions = ann.predict(TestX)

    for i in range(TestX.shape[0]):
        print predictions[i], TestY[i]
    
    print "MSE: {}".format(rmse(predictions, TestY))
    print "Corr: {}".format(ccc(TestY, predictions))
    print "R2: {}".format(r2(TestY, predictions))

#fig, ax = plt.subplot(111)
#ax.plot(train_error)
#ax.plot(val_error)

# test_outputs = zeros(TrainValX.shape[0])
# for j in range(TrainValX.shape[0]):
# 	test_outputs[j] = ann.predict(TrainValX[j])
# 	print test_outputs[j], TrainValY[j]
# print "MSE: {}".format(rmse(test_outputs, TrainValY))

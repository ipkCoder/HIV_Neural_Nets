from qsarHelpers import *
from ANN import *
import argparse
import os
import math

def main(trainDataPath, train_pIC50Path, validationDataPath, validation_pIC50Path, testDataPath, test_pIC50Path):
    try:
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = getAllOfTheData(trainDataPath, train_pIC50Path, validationDataPath, validation_pIC50Path, testDataPath, test_pIC50Path)

        # combine test and training sets
        # Backprop trainer will split it up
        TrainValX = append(TrainX, ValidateX, axis=0)
        TrainValY = append(TrainY, ValidateY, axis=0)

        # rescale data
        TrainValX, TrainValY, TestX, TestY = rescaleTheData(TrainValX, TrainValY, TestX, TestY)

        modeler = ANN(TrainValX.shape[1]);
        
        modeler.train(TrainX,TrainY);
        
        # TODO: reorganizing, roll-up any relevant code from main into seperate module(s)
        # Backprop trainer object
        #trainer = BackpropTrainer(ffn, ds)#, learningrate=.4, momentum=.2)#, verbose=True)

        # learning rates to test
        # note: best rates have been between .07 and .1, but seems to very inbetween
        # most consistent between .07 and .085
        #alpha = array([0, .05, .07, .0775, .085, .1, .2])#, .15, .2, .25, .3, 3.5])
        #momentum = array([.05, .1, .15])
        # test learning rates
        #for i in range(alpha.shape[0]):
            # randomize weights
            #ffn.randomize();
            #for k in range(momentum.shape[0]):
                # Backprop trainer object
                #trainer = BackpropTrainer(ffn, ds, learningrate=alpha[i], momentum=momentum[k])
                #, verbose=True)
                # for j in range(1000):
                # 	error = trainer.train()
                # 	if j%10 == 0:
                # 		print "{} error: {}".format(j, error)
                # 	# print "All weights: {}".format(ffn.params)
                # 	if error < .001:
                # 		break 
                # splits data into 75% training, 25% validation
                # train until convergence
                #error = trainer.trainUntilConvergence(maxEpochs=20, continueEpochs=10)# validationPortion=.X
                # print results
                #print "alpha: {}, momentum: {}".format(alpha[i], momentum[k]);
                #train_outputs = zeros(TrainValX.shape[0]);
                #for j in range(TrainValX.shape[0]):
                    #train_outputs[j] = ffn.activate(TrainValX[j]);
                    # print train_outputs[j], TrainValY[j]
                    #print "Train MSE: {}".format(Validator.MSE(train_outputs, TrainValY));
                    #test_outputs = zeros(TestX.shape[0]);
                    #for j in range(TestX.shape[0]):
                        #test_outputs[j] = ffn.activate(TestX[j]);
                        # print test_outputs[j], TestY[j]
                        #print "Test MSE: {}".format(Validator.MSE(test_outputs, TestY));
        return 0;
    except:
        print "error in main";
# ================================================================================
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(prog="pyBrain_ANN"
                                         ,description='Trains an Artificial Neural Network '\
                                             'using the pyBrain library to generate an optimal QSAR-regression model');
        parser.add_argument('-t','--trainingDataFile'
                        ,dest="trainingData"
                        ,default = ''
                        ,help="Training data file.");
        parser.add_argument('-a','--train_pIC50Path'
                        ,dest="train_pIC50"
                        ,default = ''
                        ,help="Training pIC50 data file.");
        parser.add_argument('-v','--validationDataPath'
                        ,dest="validationData"
                        ,default = ''
                        ,help="Validation data file.");
        parser.add_argument('-b','--validation_pIC50Path'
                        ,dest="validation_pIC50"
                        ,default = ''
                        ,help="Validation pIC50 data file.");
        parser.add_argument('-s','--testDataPath'
                        ,dest="testData"
                        ,default = ''
                        ,help="Test data file.");
        parser.add_argument('-c','--test_pIC50Path'
                        ,dest="test_pIC50"
                        ,default = ''
                        ,help="Test pIC50 data file.");
        args = parser.parse_args();

        inputMask = [(1  if os.path.isfile(args.trainingData) is False else 0)
                     ,(1 if os.path.isfile(args.train_pIC50) is False else 0)
                     ,(1 if os.path.isfile(args.validationData) is False else 0)
                     ,(1 if os.path.isfile(args.validation_pIC50) is False else 0)
                     ,(1 if os.path.isfile(args.testData) is False else 0)
                     ,(1 if os.path.isfile(args.test_pIC50) is False else 0)];

        if sum(inputMask) != 0:
            print "invalid input parameters"
            log("invalid input parameters", logtype.warning);
        else:
            trainDataPath        = args.trainingData;
            train_pIC50Path      = args.train_pIC50;
            validationDataPath   = args.validationData;
            validation_pIC50Path = args.validation_pIC50;
            testDataPath         = args.testData;
            test_pIC50Path       = args.test_pIC50;
            main(trainDataPath, train_pIC50Path, validationDataPath, validation_pIC50Path, testDataPath, test_pIC50Path);
    except:
        print "error in main";

import time                 # provides timing for benchmarks
from numpy   import *        # provides complex math and array functions
#from sklearn import svm        # provides Support Vector Regression
import csv
import math
import sys
import hashlib
from qsarHelpers import *

# TODO: review evaluation formulas
#         - need to document
#         - names?
#         - what are they?
#         - understand what they mean
#         - appropriate for non-linear models?

#TODO: most of the time dealing with population is spent in cv_predict
#      have to train number of models (one without each sample)
#      dealing with one population member takes about five minutes
#      about 95% of this time is spent in cv_predict
#      seems like we should be able to use CUDA for parallel programming

#------------------------------------------------------------------------------

def r2(y, yHat):
    """Coefficient of determination"""

    numer = ((y - yHat)**2).sum()       # Residual Sum of Squares
    denom = ((y - y.mean())**2).sum()   # total variation in y
    r2 = 1 - numer/denom                # all variation (minus) variation not described by model
    return r2                           # variation described by model
                                        #   - if 1, all variation described by model,
                                        #     model would be perfect fit
                                        #   - if 0, model can't explain any variation
                                        #   - in y values
#------------------------------------------------------------------------------

def r2Pred(yTrain, yTest, yHatTest):

    numer = ((yHatTest - yTest)**2).sum()
    denom = ((yTest - yTrain.mean())**2).sum()      # why this way?
    r2Pred = 1 - numer/denom
    return r2Pred

#------------------------------------------------------------------------------

def see(p, y, yHat):
    """
    Standard error of estimate
    (Root mean square error)
    """
    n = y.shape[0]
    numer = ((y - yHat)**2).sum()
    denom = n - p - 1
    if (denom == 0):
        s = 0
    elif ( (numer/denom) < 0 ):
        s = 0.001
    else:
        s = (numer/denom)** 0.5
    return s
#------------------------------------------------------------------------------

# Purpose - compute standard deviation of predictions

def sdep(y, yHat):
    """
    Standard deviation of error of prediction
    (Root mean square error of prediction)
    """
    n = y.shape[0]

    numer = ((y - yHat)**2).sum()

    sdep = (numer/n)**0.5  # square root

    return sdep
#------------------------------------------------------------------------------


def ccc(y, yHat):
    """Concordance Correlation Coefficient"""
    n = y.shape[0]
    numer = 2*(((y - y.mean())*(yHat - yHat.mean())).sum())
    denom = ((y - y.mean())**2).sum() + ((yHat - yHat.mean())**2).sum() + n*((y.mean() - yHat.mean())**2)
    ccc = numer/denom
    return ccc

#------------------------------------------------------------------------------

def ccc_adj(ccc, n, p):

    """
    Adjusted CCC
    Parameters
    ----------
    n : int -- Sample size
    p : int -- Number of parameters
    
    """
    ccc_adj = ((n - 1)*ccc - p)/(n - p - 1)
    return ccc_adj

#------------------------------------------------------------------------------

def q2F3(yTrain, yTest, yHatTest):
    numer = (((yTest - yHatTest)**2).sum())/yTest.shape[0]
    denom = (((yTrain - yTrain.mean())**2).sum())/yTrain.shape[0]
    q2F3 = 1 - numer/denom
    return q2F3
#------------------------------------------------------------------------------

def k(y, yHat):
    """Compute slopes"""
    k = ((y*yHat).sum())/((yHat**2).sum())
    kP = ((y*yHat).sum())/((y**2).sum())
    return k, kP

#------------------------------------------------------------------------------

def r0(y, yHat, k, kP):
    """
    Compute correlation for regression lines through the origin
    Parameters
    ----------
    k  : float -- Slope
    kP : float -- Slope
    
    """
    numer = ((yHat - k*yHat)**2).sum()
    denom = ((yHat - yHat.mean())**2).sum()
    r2_0 = 1 - numer/denom
    numer = ((y - kP*y)**2).sum()
    denom = ((y - y.mean())**2).sum()
    rP2_0 = 1 - numer/denom
    return r2_0, rP2_0

#------------------------------------------------------------------------------

def r2m(r2, r20):
    """Roy Validation Metrics"""
    r2m = r2*(1 - (r2 - r20)**0.5)
    return r2m

#------------------------------------------------------------------------------

def r2m_adj(r2m, n, p):

    """
    Adjusted r2m 
    Parameters
    ----------
    n : int -- Number of observations
    p : int -- Number of predictor variables
    
    """
    r2m_adj = ((n - 1)*r2m - p)/(n - p - 1)
    return r2m_adj

#------------------------------------------------------------------------------

def r2p(r2, r2r):
    """
    Parameters
    ----------
    r2r : float --Average r^2 of y-randomized models.
    
    """
    r2p = r2*((r2 - r2r)**0.5)
    return r2p

#------------------------------------------------------------------------------

def rmse(X, Y):
    """
    Calculate the root-mean-square error (RMSE) also known as root mean
    square deviation (RMSD).
    
    Parameters
    ----------
    X : array_like -- Assumed to be 1D.
    Y : array_like -- Assumed to be the same shape as X.
    
    Returns
    -------
    out : float64
    """
    X = asarray(X, dtype=float64)
    Y = asarray(Y, dtype=float64)
    return (sum((X-Y)**2)/len(X))**.5
#------------------------------------------------------------------------------
"""
    Purpose: - cross validate training data
             - for each sample, train model without that sample, then predict target for that sample
    
    @param set_x: feature set for training set data
    @param set_y: target values for training set data
    @param val_x: feature set for validation set data
    @param val_y: target values for validation set data

    @variable yhat: array to hold predicted values for each training sample (size same as set_x)
    @variable train_x: training set without sample to predict
    @variable train_y: target values for training set without target value for sample to predict
    @variable model_name: name of model used to predict values
    @variable idx: index of sample/target value to drop from set_x/set_y

    note: val_x and val_y are used in training model because pyBrain requires validation data
"""

def cv_predict(set_x, set_y, val_x, val_y, model):
    """Predict using cross validation."""
    
    #print "Models to train: {}".format(set_x.shape[0])

    yhat = empty_like(set_y)
    
    for idx in range(0, yhat.shape[0]):
        train_x = delete(set_x, idx, axis=0)
        train_y = delete(set_y, idx, axis=0)
        try:
            with Timer() as t:
                modelName = model.train(train_x, train_y, val_x, val_y)
                yhat[idx] = model.predict(set_x[idx])
        except:
            print "Error with training cv model for sample {}".format(idx)
        #finally:
            #print("Trained cv individual {} in cv_predict: {:.03f} sec.".format( idx,t.interval))
    return yhat
#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def calc_fitness(xi, Y, Yhat, c=2):
    """
    Calculate fitness of a prediction.
    Parameters
    ----------
    xi : array_like -- Mask of features to measure fitness of. Must be of dtype bool.
    model : object  --  Object to make predictions, usually a regression model object.
    c : float       -- Adjustment parameter.
    
    Returns
    -------
    out: float -- Fitness for the given data.
    
    """

    p     = len(xi);
    n     = len(Y)    # Sample size
    numer = ((Y - Yhat)**2).sum()/n   # Mean square error
    pcn   = p*(c/n)
    if pcn >= 1:
        return 1000
    denom = (1 - pcn)**2
    theFitness = numer/denom
    return theFitness
#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def InitializeTracks():
    trackDesc = {}
    trackIdx = {}
    trackFitness = {}
    trackModel = {}
    trackR2 = {}
    trackQ2 = {}
    trackR2PredValidation = {}
    trackR2PredTest = {}
    trackSDEPTrain = {}
    trackSDEPValidation = {}
    trackSDEPTest = {}
    return  trackDesc, trackIdx, trackFitness, trackModel, trackR2, trackQ2, \
            trackR2PredValidation, trackR2PredTest, trackSDEPTrain, \
            trackSDEPValidation, trackSDEPTest
#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def initializeYDimension():
    yTrain = {}
    yHatTrain = {}
    yHatCV = {}
    yValidation = {}
    yHatValidation = {}
    yTest = {}
    yHatTest = {}
    return yTrain, yHatTrain, yHatCV, yValidation, yHatValidation, yTest, yHatTest 
#------------------------------------------------------------------------------
def OnlySelectTheOnesColumns(popI):
    
    indicies_of_non_zeros = popI.nonzero()[0]
    indicies_of_non_zeros.tolist()

    return indicies_of_non_zeros

#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def validate_model(generation, model, fileW, population, TrainX, TrainY,\
                   ValidateX, ValidateY, TestX, TestY):
    numOfPop   = population.shape[0]
    fitness    = zeros(numOfPop)
    c          = 2
    false      = 0
    true       = 1
    predictive = false
  
    # create empty dictionaries to hold data
    trackDesc, trackIdx, trackFitness,trackModel,trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest, trackSDEPTrain, \
    trackSDEPValidation, trackSDEPTest = InitializeTracks()
    
    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest    = initializeYDimension()
    
    unfit                              = 1000
    itFits                             = 1
    # analyze one population row at a time
    for i in range(numOfPop):
        with Timer() as t:
            error = False
            # try attempt fails
            # Suggestion: check for and update local pyBrain libraries
            try:
                with Timer() as t0:
                    xi       = OnlySelectTheOnesColumns(population[i])
                    idx      = hashlib.sha1(array(xi)).digest() # Hash
                    if idx in trackFitness.keys():
                        # don't recalculate everything if the model has already been validated
                        fitness[i] = trackFitness[idx]
                        continue
                    # get columns that have been selected for evaluation
                    X_train_masked      = TrainX.T[xi].T
                    X_validation_masked = ValidateX.T[xi].T
                    X_test_masked       = TestX.T[xi].T

                    model.create_network(X_train_masked.shape[1])
                    model_desc = model.train(X_train_masked, TrainY, X_validation_masked, ValidateY)
            except:
                print "failed to train"
                error = True
            finally:
                print "Trained model for population {} in {} sec".format(i, t0.interval)
                if error:
                    return unfit, fitness
            ''' 
            # Computed cross validation predictions MARK: cv
            try:
                with Timer() as t0:
                    pass#Yhat_cv = cv_predict(X_train_masked, TrainY, X_validation_masked, ValidateY, model)    # Cross Validation
            finally:
                print "Cross validate for {}: {}".format(i, t0.interval)
            '''
            # predict validation and test data using trained model
            Yhat_train      = model.predict(X_train_masked)
            Yhat_validation = model.predict(X_validation_masked)
            Yhat_test       = model.predict(X_test_masked)
            
            # Compute statistics for the coefficient of determination (R2)
            # i.e. Prediction for Validation and Test set
            r2_train          = r2(TrainY, Yhat_train)
       #---     q2_loo            = r2(TrainY, Yhat_cv)                              # MARK: cv
            r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
            r2pred_test       = r2Pred(TrainY, TestY, Yhat_test)
            
            # calculate fitness, use training and validation prediction/target values
            Y_fitness         = append(TrainY, ValidateY)
	    #Y_fitness         = append(Y_fitness, TestY)
	    Y_fitness         = append(Y_fitness, TestY)
       #---     Yhat_fitness      = append(Yhat_cv, Yhat_validation)               # MARK: cv
            Yhat_fitness      = append(Yhat_train, Yhat_validation)
            #Yhat_fitness      = append(Yhat_fitness, Yhat_test)               # MARK: cv
            Yhat_fitness      = append(Yhat_fitness, Yhat_test)               # MARK: cv
	    fitness[i]        = calc_fitness(xi, Y_fitness, Yhat_fitness, c)
            
            # ignore and continue if predictive quality is too low
            # between either the training, cross-validation or test sets
            # i.e. if it's not worth recording, just return the fitness
            if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
                print "ending the program, prediction is : ", predictive
                continue;
            
            # Standard error of estimate (RMSE)
            sdep_train                 = sdep(TrainY, Yhat_train)
            sdep_validation            = sdep(ValidateY, Yhat_validation)
            sdep_test                  = sdep(TestY, Yhat_test)
            
            # store stats
            idxLength                  = len(xi)

            trackDesc[idx]             = str(xi)
            trackIdx[idx]              = idxLength
            trackFitness[idx]          = fitness[i]
            trackModel[idx]            = model_desc
            
       #---     trackQ2[idx]               = q2_loo             # MARK: cv
            
            trackR2[idx]               = r2_train
            trackR2PredValidation[idx] = r2pred_validation
            trackR2PredTest[idx]       = r2pred_test
            
            trackSDEPTrain[idx]        = sdep_train
            trackSDEPValidation[idx]   = sdep_validation
            trackSDEPTest[idx]         = sdep_test
            
            yTrain[idx]                = TrainY.tolist()
            yHatTrain[idx]             = Yhat_train.tolist()
       #---     yHatCV[idx]                = Yhat_cv.tolist()           # MARK: cv
            yValidation[idx]           = ValidateY.tolist()
            yHatValidation[idx]        = Yhat_validation.tolist()
            yTest[idx]                 = TestY.tolist()
            yHatTest[idx]              = Yhat_test.tolist()
        
            inToHiddenParams, inToOutParams        = model.getParams()

        print "Trained and found results for population {}: {:.03} sec".format(i, t.interval)

        #printing the information into the file MARK: cv
        '''write(fileW, trackModel[idx], generation, i, trackDesc[idx], trackIdx[idx], trackFitness[idx],
            trackR2[idx], trackQ2[idx],trackR2PredValidation[idx], trackR2PredTest[idx], trackSDEPTrain[idx], \
            trackSDEPValidation[idx], trackSDEPTest[idx], yTrain[idx], yHatTrain[idx], yHatCV[idx], \
            yValidation[idx], yHatValidation[idx], yTest[idx], yHatTest[idx], inToHiddenParams, inToOutParams)
        '''
        write(fileW, trackModel[idx], generation, i, trackDesc[idx], trackIdx[idx], trackFitness[idx],
            trackR2[idx],trackR2PredValidation[idx], trackR2PredTest[idx], trackSDEPTrain[idx], \
            trackSDEPValidation[idx], trackSDEPTest[idx], yTrain[idx], yHatTrain[idx], \
            yValidation[idx], yHatValidation[idx], yTest[idx], yHatTest[idx], inToHiddenParams, inToOutParams)
    
    return itFits, fitness

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
#Ahmad Hadaegh: Modified  on: July 16, 2013 MARK: cv
def write(fileW, model, generation, individual, trackDesc, trackIdx, trackFitness, trackR2, \
          trackR2PredValidation, trackR2PredTest, trackSDEPTrain, \
          trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain,  \
          yValidation, yHatValidation, yTest, yHatTest, inToHiddenParams, inToOutParams): 

    # MARK: cv
    fileW.writerow([model, generation, individual, trackDesc, trackIdx, trackFitness, trackR2, \
        trackR2PredValidation, trackR2PredTest, trackSDEPTrain, \
        trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain,  \
        yValidation, yHatValidation, yTest, yHatTest, inToHiddenParams, inToOutParams])
    #fileOut.close()
'''
#Ahmad Hadaegh: Modified  on: July 16, 2013 MARK: cv
def write(fileW, model, generation, individual, trackDesc, trackIdx, trackFitness, trackR2, \
          trackR2PredValidation, trackR2PredTest, trackSDEPTrain, \
          trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain, yHatCV, \
          yValidation, yHatValidation, yTest, yHatTest, inToHiddenParams, inToOutParams): 

    # MARK: cv
    fileW.writerow([model, generation, individual, trackDesc, trackIdx, trackFitness, trackR2, \
        trackR2PredValidation, trackR2PredTest, trackSDEPTrain, \
        trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain, yHatCV, \
        yValidation, yHatValidation, yTest, yHatTest, inToHiddenParams, inToOutParams])
    #fileOut.close()
'''

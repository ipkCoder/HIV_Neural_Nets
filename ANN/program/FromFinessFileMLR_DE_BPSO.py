import time                 # provides timing for benchmarks
from numpy   import *        # provides complex math and array functions
#from sklearn import svm	    # provides Support Vector Regression
import csv
import math
import sys
import hashlib

#------------------------------------------------------------------------------


def r2(y, yHat):
    """Coefficient of determination"""
    numer = ((y - yHat)**2).sum()       # Residual Sum of Squares
    denom = ((y - y.mean())**2).sum()
    r2 = 1 - numer/denom
    return r2
#------------------------------------------------------------------------------

def r2Pred(yTrain, yTest, yHatTest):
    numer = ((yHatTest - yTest)**2).sum()
    denom = ((yTest - yTrain.mean())**2).sum()
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
    elif ( (numer/denom) <0 ):
        s = 0.001
    else:
        s = (numer/denom)** 0.5
    return s
#------------------------------------------------------------------------------

def sdep(y, yHat):
    """
    Standard deviation of error of prediction
    (Root mean square error of prediction)
    """
    n = y.shape[0]

    numer = ((y - yHat)**2).sum()

    sdep = (numer/n)**0.5

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

def rSquared(y, yPred):
	"""Find the coefficient  of correlation for an actual and predicted set.
	
	Parameters
	----------
	y : 1D array -- Actual values.
	yPred : 1D array -- Predicted values.
	
	Returns 
	-------
	out : float -- Coefficient  of correlation.
	
	"""

	rss = ((y - yPred)**2).sum()   # Residual Sum of Squares
	sst = ((y - y.mean())**2).sum()   # Total Sum of Squares
	r2 = 1 - (rss/sst)

	return r2

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

def cv_predict(set_x, set_y, model):
    """Predict using cross validation."""
    yhat = empty_like(set_y)
    for idx in range(0, yhat.shape[0]):
        train_x = delete(set_x, idx, axis=0)
        train_y = delete(set_y, idx, axis=0)
        modelName = model.train(train_x, train_y)
        yhat[idx] = model.predict(set_x[idx])
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
    
#    p = sum(populationRowI[xi])   # Number of selected parameters (sum value of indexes ????)
    p = len(xi);
    n = len(Y)    # Sample size
    numer = ((Y - Yhat)**2).sum()/n   # Mean square error
    pcn = p*(c/n)
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
    trackSEETrain = {}
    trackSDEPValidation = {}
    trackSDEPTest = {}
    return  trackDesc, trackIdx, trackFitness, trackModel, trackR2, trackQ2, \
            trackR2PredValidation, trackR2PredTest, trackSEETrain, \
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
    numOfFea = popI.shape[0]
    xi = zeros(numOfFea)
    for j in range(numOfFea):
       xi[j] = popI[j]

    xi = xi.nonzero()[0]
    xi = xi.tolist()
    return xi
 
#------------------------------------------------------------------------------

#Ahmad Hadaegh: Modified  on: July 16, 2013
def validate_model(model, fileW, population, TrainX, TrainY,\
                   ValidateX, ValidateY, TestX, TestY):
    
    numOfPop = population.shape[0]
    fitness = zeros(numOfPop)
    c = 2
    false = 0
    true = 1
    predictive = false
  
    # create empty dictionaries to hold data
    trackDesc, trackIdx, trackFitness,trackModel,trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest, trackSEETrain, \
    trackSDEPValidation, trackSDEPTest = InitializeTracks()

    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest = initializeYDimension()

    unfit = 1000
    itFits = 1
    # analyze one population row at a time
    for i in range(numOfPop):
        
        xi = OnlySelectTheOnesColumns(population[i])
      
        idx = hashlib.sha1(array(xi)).digest() # Hash
      
        if idx in trackFitness.keys():
            # don't recalculate everything if the model has already been validated
            fitness[i] = trackFitness[idx]
            continue
      
        # get columns that have been selected for evaluation
        X_train_masked = TrainX.T[xi].T
        X_validation_masked = ValidateX.T[xi].T
        X_test_masked = TestX.T[xi].T
  
        model.create_network(X_train_masked.shape[1], 20, 1)

        print model.get_params()
  
        try:
            model_desc = model.train(X_train_masked, TrainY, X_validation_masked, ValidateY)
            print "finished training"
        except:
            print "didn't train"
            return unfit, fitness

        
        # Computed predicted values using computed y-int and slope 
        Yhat_cv = cv_predict(X_train_masked, TrainY, model)    # Cross Validation
        Yhat_validation = model.predict(X_validation_masked)
        Yhat_test = model.predict(X_test_masked)
            
        # Compute R2 statistics (Prediction for Valiation and Test set)
        q2_loo = r2(TrainY, Yhat_cv)
        r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
        r2pred_test = r2Pred(TrainY, TestY, Yhat_test)
                      
        Y_fitness = append(TrainY, ValidateY)
        Yhat_fitness = append(Yhat_cv, Yhat_validation)

        fitness[i] = calc_fitness(xi, Y_fitness, Yhat_fitness, c)
        
        #print "predictive is: ", predictive
        if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
            # if it's not worth recording, just return the fitness
            print "ending the program because of predictive is: ", predictive
            continue
            
        # Compute predicted Y_hat for training set.
        Yhat_train = model.predict(X_train_masked)
        r2_train = r2(TrainY, Yhat_train)
   
        # Standard error of estimate
        s = see(X_train_masked.shape[1], TrainY, Yhat_train)
        sdep_validation = sdep(ValidateY, Yhat_validation)
        sdep_test = sdep(TrainY, Yhat_train)
            
        idxLength = len(xi)

        # store stats
        trackDesc[idx] = str(xi)
        trackIdx[idx] = idxLength
        trackFitness[idx] = fitness[i]

        trackModel[idx] = model_desc
        
        trackR2[idx] = r2_train
        trackQ2[idx] = q2_loo
        trackR2PredValidation[idx] = r2pred_validation
        trackR2PredTest[idx] = r2pred_test
        trackSEETrain[idx] = s
        trackSDEPValidation[idx] = sdep_validation
        trackSDEPTest[idx] = sdep_test

        yTrain[idx] = TrainY.tolist()
        yHatTrain[idx] = Yhat_train.tolist()
        yHatCV[idx] = Yhat_cv.tolist()
        yValidation[idx] = ValidateY.tolist()
        yHatValidation[idx] = Yhat_validation.tolist()
        yTest[idx] = TestY.tolist()
        yHatTest[idx] = Yhat_test.tolist()
        
        
    #printing the information into the file

    write(model,fileW, trackDesc, trackIdx, trackFitness, trackModel, trackR2,\
                trackQ2,trackR2PredValidation, trackR2PredTest, trackSEETrain, \
                trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain, yHatCV, \
                yValidation, yHatValidation, yTest, yHatTest)
    
    return itFits, fitness
#------------------------------------------------------------------------------  
#Ahmad Hadaegh: Modified  on: July 16, 2013

def write(model,fileW, trackDesc, trackIdx, trackFitness, trackModel, trackR2, \
          trackQ2,trackR2PredValidation, trackR2PredTest, trackSEETrain, \
          trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain, yHatCV, \
          yValidation, yHatValidation, yTest, yHatTest): 
    
    for key in trackFitness.keys():
        fileW.writerow([trackDesc[key], trackIdx[key], trackFitness[key], trackModel[key], \
            trackR2[key], trackQ2[key], trackR2PredValidation[key], trackR2PredTest[key], \
            trackSEETrain[key], trackSDEPValidation[key], trackSDEPTest[key], \
            yTrain[key], yHatTrain[key], yHatCV[key], yValidation[key], yHatValidation[key], \
            yTest[key], yHatTest[key]])
    #fileOut.close()

#------------------------------------------------------------------------------


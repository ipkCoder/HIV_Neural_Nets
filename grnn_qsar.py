#-------------------------------------------------------------------------------
# Name:        Generalized Regression Neural Networks (GRNN)
# Description: A prototype trainer for a GRNN-QSAR modeler
#
# Author:      CSUSM-CS-HIV_Research-Group_1
#
# Created:     20140608
# Copyright:   (c) 2014
# Licence:     MIT
#-------------------------------------------------------------------------------
import numpy as np
import math


# ======= Regarding first TODO ===============
# what do you mean by map input layer to pattern layer?
# each trainging sample does not get its own sigma values

# we really dont need to keep track of best and work y-hat
# i dont remember why i did that, I think just to see if at end best was better

# for cost function
#   - are you wanting to calculate cost of each sample one at a time (through function getE)?

#activation function should include exp(-D)
#not understanding input layer/input matrix

# outputLayer function only returns one y-hat (for example idx)
# y should be y[x_i] in function getCosts
# need to randomly initialize sigmas


# ================================================================================

# x1 - x sample vector for which to compute predicted value
# x2 - other x sample in set samples
# sigma - vector of weights (each weight corresponds to specific feature)
# return D - squared weighted euclidean distance (single value)

def activationFunction(x1, x2, sigma):
  '''Get the squared weighted Euclidean distance'''
  try:
    D = sum( ( float(x1-x2) / float(sigma) )**2);
    return D;
  except:
    print "Error calculating squared weighted Euclidean Distance"
# ================================================================================

# y_i - actual y value for sample i
# D - squared weighted euclidean distance
# out - activation function value

# note: exp(- is part of activation function (not part of summation layer)
# should move exp(- into activation function above

def getSummationLayerNumerator(y_i,D):
  '''Get the summation layer numerator (see eq.6 / 11)'''
  try:
    out = float(y_i*math.exp(-D));
    return out;
  except:
    print "Error calculating summation layer numerator"
# ================================================================================

# transform D (should be part of activation function
# note: denominator is just the summation of the activation function values
#             - activation function values are the weights of each y value

def getSummationLayerDenominator(D):
  '''Get the summation layer denominator (see eq.6 / 11)'''
  try:
    return float(math.exp(-D));
  except:
    print "Error getting summation layer denominator."
# ================================================================================

# this code is only computing the predicted value of sample number idx
#  - not a vector of all y-hat values

# what do you want to do here?
#    - find vector of y-hats or single y-hat?
#    - your comment says vector of y-hats

# this function needs to have access to the sample x
#     - if XminusXI is passed in as X, then sample x is not present in function
#     - if X is all Xs, then need to test to make sure X[idx] != X[i] in loop

#activation function needs to have access to all sigmas, not just one
#    - sigma[i] will cause problem, # of Xs not the same as number of sigmas
#    - problem solved by removing [i] and passing in sigma vector

def outputLayer(idx, X, y, sigma):
  '''Summation layer "...generates the vector of predicted
    y-values.", i.e. Y-hat'''
  try:
    num   = 0;
    denom = 0;
    # find the summation layer numerator(s) and denominator(s)
    for i in range(X.shape[0]):
      D      = activationFunction(X[idx],X[i],sigma[i]);
      num   += getSummationLayerNumerator(y[i],D);
      denom += getSummationLayerDenominator(D);
    # return predicted y-values, i.e. y-hat
    return (float(num)/float(denom));
  except:
    print "Error predicting Y-hat in summation layer";
# ================================================================================

# y - is y a vector or single value?
# you convert out to a float twice (reason?)

def getE(idx, X, y, sigma):
  '''Cost function component'''
  try:
    bX  = outputLayer(idx, X, y, sigma);
    out = float( (bX - y)**2 );
    e   = float(out);
    return e;
  except:
    print "Error getting...error :)";
# ================================================================================

# could just divide by N at the end instead of in each loop
# in second loop, should be += cost
# copying who matrix except for one row seems costly
#    - better to check if the two samples are the same
#         - if they are, just skip over

def getCosts(X,y,sigma):
  '''For each member of the training set, computes the cost of misclassification.'''
  try:
    costs = np.zeros(len(X));
    # for each training set member (i.e. x_i )
    for xI in X:
      
      num = 0;
      denom = N = len(X);
      # find error of predicted value for sample
      # derive from other samples (exclude comparison with self)
      for x_i in xrange(0,N):
        
        if (X[idx] == X[x_i]).all()v == false:
          
          num += getE(X[idx], X[x_i], y[x_i], sigma);
          costs[xI] += float(num) / float(denom);
    
    return costs;
  
  except:
    print "Error computing cost";
# ================================================================================
# main function
def main():
  try:
    # unlabeled training set (X) to classify
    X = np.array([[1.0,2.0,3.0,4.0,5.0],
                  [1.0,1.0,1.0,1.0,1.0],
                  [2.0,3.0,1.5,2.0,1.9],
                  [1.5,2.5,2.0,2.0,2.5]]);
    
    # observed target y
    y = np.array([6.0,1.3,2.4,1.8,2.2]);
    
    # array to hold predicted values
    yHat = np.zeros(y.shape[0]);
    
    # Multiple-sigma model, one weight per descriptor
    sigmas     = np.arange(1.0,6.0);
    # TODO: How to map trainig sample input layer(s) to pattern layer(s)?
    # Notes: From paper, for a single instance, feed each
    # descriptor feature / sigma weight pair of each training sample
    # to each node in the pattern layer.
    # In general, perform this operation in parallel for every
    # sample in the training set.  Verify.
    
    # node in pattern layer for each training set member
    patternLayer = [];
    for patternNode in xrange(0,len(X)):
      inputMatrix = [];
      # generate input layer matrix for each pattern node
      for idx in xrange(0,len(X)):
        inputLayer = np.dstack((X[idx],sigmas));
        inputMatrix.append(inputLayer);
    
    #TODO: append input matrix to pattern node
    # repeat for each training sample member
    # print inputMatrix
    # best and worst set of predicted values
    # determined by cost function
    bestYHat  = np.zeros(y.shape[0])
    worstYHat = np.zeros(y.shape[0])
    
    # best and worst starting values
    bestCost  = 100000
    worstCost = 0
    
    return 0;
  except:
    print "Error in main";
# ================================================================================
if __name__ == '__main__':
  main();
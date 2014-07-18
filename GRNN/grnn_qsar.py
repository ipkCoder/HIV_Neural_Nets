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
from ImportData import *

# ======= Regarding first TODO ===============
# what do you mean by map input layer to pattern layer?
# each training sample does not get its own sigma values
# R: Hi Ian, I'm just trying to understand how data passes from "input" layer(s)
# to the "pattern" layer(s).  I'm going off the Fig. 3 from the paper.
# I guess what I'm not completely understanding is how the "sigma" weight(s) vector is applied
# to the training sample(s) in the "input" layer stage.
# Ian: In the input layer, nothing is done to the features, they just enter the network
# similar to ANN

# we really dont need to keep track of best and work y-hat
# I dont remember why i did that, I think just to see if at end best was better
# R: We can remove these if not needed, I hadn't gotten that far in revising the code
# so I left those variables as-is since I wan't sure if they were relevant.

# for cost function
#   - are you wanting to calculate cost of each sample one at a time (through function getE)?
# R: cost function is not being utilized at this point, but yes the intention is that 
# we would be using this to identify the error, i.e. cost of misclassifying a
# training sample instance.

#activation function should include exp(-D)
#not understanding input layer/input matrix
# R: this ties in to my attempt to understand how the input layer => pattern layer works.
# My thinking was that each "pattern" node gets a reference too all the training samples, including
# the feature / sigma weight pairs.

# Ian: all nodes in the pattern layer do not get a reference to all training sample data.
# Each node in the pattern layer contains the features for one sample (the number of nodes
# in the patern layer is equal to the number of samples). Each node in the pattern layer computes
# the squared weighted euclidean distance between the input layer sample and the sample in the node
# and the activation function value. 

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
# R: "y" is meant to be a vector in this context, this code was part of another function previously
# I just rolled it up into a standalone function. No need for the float
# good catch.  Note, if you see an obvious mistake, feel free to update the source, that's
# the idea behind collaborative revision control.  Contributing includes
# making adding / removing / updating the source.
def getE(idx, X, y, sigma):
  '''Cost function component'''
  try:
    bX  = outputLayer(idx, X, y, sigma);
    out = float( (bX - y)**2 );
    return out;
  except:
    print "Error getting...classification error.";
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
    
	data, targets = getAllOfTheData()

	sample_size, feature_size = data.shape

	# unlabeled training set (X) to classify
    X = np.array([[1.0,2.0,3.0,4.0,5.0],
                  [1.0,1.0,1.0,1.0,1.0],
                  [2.0,3.0,1.5,2.0,1.9],
                  [1.5,2.5,2.0,2.0,2.5]]);
    
    # observed target y
    y = np.array([6.0,1.3,2.4,1.8,2.2]);
    
    # array to hold predicted values
    yHat = np.zeros(sample_size);
    
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

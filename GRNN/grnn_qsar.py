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
import sys

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
# x2 - other x sample in dataset
# sigma - vector of weights (each weight corresponds to specific feature)
# return D - squared weighted euclidean distance (single value)

# TODO: make sure sigma isn't zero, if is, ???
def weightedEuclideanDistance(x1, x2, sigma):
    '''Get the squared weighted Euclidean distance between vector x1 and x2'''
    #print len(x1)
    #print len(x2)
    #print len(sigmq)
    
    try:
        squared_weighted_dist = ((x1-x2)/sigma)**2
#        print squared_weighted_dist
        squared_weighted_dist[squared_weighted_dist == np.inf] = 10000000
#        print squared_weighted_dist[squared_weighted_dist == 1000000]
        distance = sum(squared_weighted_dist)
        return distance;
    except:
        print "Error calculating squared weighted Euclidean Distance"

# compute disctances between all samples in dataset
# @local euclidean_distance - type matrix
#   - euclidean_distance[i][j] : square weighted euclidean distance between sample_i and sample_j
def computeDistances(data, sigma):

    sample_size = data.shape[0]

    euclidean_distance = np.zeros((sample_size, sample_size))

    for i in range(sample_size):
        for j in range(sample_size):
#           print i, j
#           sys.stdout.flush()
            try:
                euclidean_distance[i][j] = weightedEuclideanDistance(data[i], data[j], sigma)
#                if euclidean_distance[i][j] == 0:
#                    print i, j
            except:
                print("error calculating distances between {0} and {1}".format(i, j))

    return euclidean_distance
# ================================================================================

# @param y          - (vector) actual y values
# @param distance   - (matrix) squared weighted euclidean distance between samples
# @return numerator - (vector) nummerator of predicted y values

def getSummationLayerNumerator(y, distance):
    '''Get the summation layer numerator (see eq.6 / 11)'''
    try:
        # multiply y vector along x axis
        numerator = sum(y * np.exp(-distance).T)
        return numerator
    except:
        print "Error calculating summation layer numerator"
# ================================================================================

# @param distance - (matrix) squared weighted euclidean distance between samples
# @return           (vector) denominator of predicted y values

def getSummationLayerDenominator(distance):
    '''Get the summation layer denominator (see eq.6 / 11)'''
    try:
        return sum(np.exp(-distance));
    except:
        print "Error getting summation layer denominator."
# ================================================================================

# @param numerator   - (vector) numerator from summation layer
# @param denominator - (vector) denominator from summation layer
# @return      - (vector) predicted y values for each sample
def outputLayer(numerator, denominator):
    '''Divide numerator and denominator from summation layer'''
    try:
        return numerator/denominator
    except:
        print "Error predicting Y-hats in summation layer";
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
        
        if (X[idx] == X[x_i]).all() == false:
          
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

    data = rescaleTheData(data)

    sample_size, feature_size = data.shape

    print(sample_size, feature_size)

    # set initial sigmas
    sigmas = np.random.randint(100, 1000, feature_size)
    #sigmas = np.random.random(feature_size);

#    print sigmas

    assert len(sigmas) == feature_size

    distance = computeDistances(data, sigmas)

    print distance

    numerator   = getSummationLayerNumerator(targets, distance)
    print numerator
    denominator = getSummationLayerDenominator(distance)
    print denominator
    yHat = outputLayer(numerator, denominator)
    print len(yHat)
#    exp_dist = np.exp(-distance)
#    print exp_dist

    for i in range(sample_size):
        print i, targets[i], yHat[i]


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
    sigmas     = np.arange(1.0,sample_size+1);

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

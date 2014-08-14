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
import argparse
import random
import math
from ImportData import *
import sys

# ======= Regarding TODOs ===============
# =========== Notes =====================
# To launch program, provide min and max sigmas
#    - ex: python grnn_qsar.py 2 100

# Prediction of target values are prefect, need to figure out why
# When change sigmas to a different range, numerators and denominators adjust
# Even with different sigmas, predictions remain perfect

# =======================================
# Purpose: find weighted euclidean distance between two samples
# @param x1        - sample 1
# @param x2        - sample 2
# @param sigma     - vector of weights (each weight corresponds to specific feature)
# @return distance - squared weighted euclidean distance (single value)
def squareWeightedEuclideanDistance(x1, x2, sigma):
    '''Get the squared weighted Euclidean distance between all samples'''
    # should assert that x1, x2, sigma of are same length
    try:
        squared_weighted_dist = ((x1-x2)/sigma)**2

        # if one of sigmas was super small, may have caused one of distances 
        # between features to become infinity. If so, just set to big number
        squared_weighted_dist[squared_weighted_dist == np.inf] = 10000000
        
        distance = sum(squared_weighted_dist)
        
        return distance;
    
    except:
        print "Error calculating squared weighted Euclidean Distance"
# ==============================================================================

# Purpose: compute disctances between all samples in dataset
# @local euclidean_distance - (matrix)
#   - euclidean_distance[i][j] : square weighted euclidean distance between sample_i and sample_j
def computeDistances(data, sigma):

    sample_size = data.shape[0]

    euclidean_distance = np.zeros((sample_size, sample_size))

    for i in range(sample_size):
        for j in range(sample_size):
            try:
                # Don't need to take euclidean_distance from yourself
                if np.array_equiv(data[i],data[j]):
                    continue;
                else:
                    euclidean_distance[i][j] = squareWeightedEuclideanDistance(data[i], data[j], sigma)
            except:
                print("Error calculating distances between {0} and {1}".format(i, j))

    return euclidean_distance
# ================================================================================

# Purpose: calculate numerators of all predicted target values

# @param y          - (vector) actual y values
# @param distance   - (matrix) squared weighted euclidean distance between samples
# @return numerator - (vector) nummerator of predicted y values

def getSummationLayerNumerator(y, distance):
    '''Get the summation layer numerator (see eq.6 / 11)'''
    
    assert len(y) == len(distance), "Number of y values should equal number of samples"

    try:
        # multiply y vector along x axis
        numerator = sum(y * np.exp(-distance))
        return numerator
    except:
        print "Error calculating summation layer numerator"
# ================================================================================

# Purpose: calculate denominators of all predicted target values

# @param distance - (matrix) squared weighted euclidean distance between samples
# @return         - (vector) denominator of predicted y values

def getSummationLayerDenominator(distance):
    '''Get the summation layer denominator (see eq.6 / 11)'''
    try:
        return sum(np.exp(-distance));
    except:
        print "Error getting summation layer denominator."
# ================================================================================

# Purpose: calculate predicted target values

# @param numerator   - (vector) numerator from summation layer
# @param denominator - (vector) denominator from summation layer
# @return            - (vector) predicted y values for each sample
#                      - for each y, predicted y is weighted sum of all ys

def outputLayer(numerator, denominator):
    '''Divide numerator and denominator from summation layer'''
    try:
        return numerator/denominator
    except:
        print "Error predicting Y-hats in output layer";
# ================================================================================

def makePredictions(distances, target_y):
    
    # iterate over all 'x' observation's distances from each other observation 'x'
    yHat = np.zeros(len(target_y))
    #print("Observation 'x' ID\tActual\tPredicted")
    for idx,observation in enumerate(distances):
        numerator   = getSummationLayerNumerator(target_y, observation)
        denominator = getSummationLayerDenominator(observation)
        yHat[idx] = numerator/denominator

    return yHat

# ================================================================================

def calcGradient(orig_sigmas, data, target_y):
    
    try:
        sigmas = np.copy(orig_sigmas).astype(float)
    
        num_sigmas = sigmas.shape[0]

        partials = np.zeros(num_sigmas)
        
        for i in range(num_sigmas):
            
            sigmas[i] *= 1.000001
            distances = computeDistances(data, sigmas)
            predictions = makePredictions(distances, target_y)
            diff = predictions - target_y
            sigma_diff = sigmas[i] - orig_sigmas[i]
            partials[i] = (diff/sigma_diff).sum()
            sigmas[i] = orig_sigmas[i] # reset sigma to orig_sigma value        
            print partials[i]
        return partials
    except:
        print("Error calculating gradient")
        exit(0)
# ================================================================================
# y - is y a vector or single value?
# you convert out to a float twice (reason?)
# R: "y" is meant to be a vector in this context, this code was part of another function previously

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
      
    parser = argparse.ArgumentParser()
    parser.add_argument("min", type=int, help="min random sigma")
    parser.add_argument("max", type=int, help="max random sigma")
    args = parser.parse_args()

    data, target_y = getAllOfTheData()

    data = rescaleTheData(data)

    #n = 20
    #data = data[:n]
    #targets = targets[:n]

    #index = random.sample(range(385), 385)
    #print index
    #data = data.T[index, :].T

    sample_size, feature_size = data.shape

    print(sample_size, feature_size)

    min_sigma = 1
    max_sigma = 1000

    # set initial sigmas
    sigmas = np.random.randint(args.min, args.max, feature_size)
    #sigmas = np.random.random(feature_size);
    #print sigmas
    
    print("Min and max sigmas: {}, {}".format(args.min, args.max))

    assert len(sigmas) == feature_size

    distances = computeDistances(data, sigmas)

    #print("Distances: ")
    #print distance

    # iterate over all 'x' observation's distances from each other observation 'x'
    yHat = []
    print("Observation 'x' ID\tActual\tPredicted")
    for idx,observation in enumerate(distances):
        numerator   = getSummationLayerNumerator(target_y, observation)
        denominator = getSummationLayerDenominator(observation)
        yHat.append(numerator/denominator)
        print("{}\t{:.04}\t{:.04}".format(idx, target_y[idx], yHat[idx]))

    print makePredictions(distances, target_y)
    print calcGradient(sigmas, data, target_y)    
    #numerator   = getSummationLayerNumerator(target_y, distances)
    #denominator = getSummationLayerDenominator(distances)

    #print("Numerators:")
    #print numerator
    #print("Denominators:")
    #print denominator
    
    #yHat = numerator/denominator #outputLayer(numerator, denominator)

    #print("  Actual\tPredicted")
    #for i in range(sample_size/5):
        #print("{}, {:.04}, {}".format(i, target_y[i], yHat[i]))

# =============================================================
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









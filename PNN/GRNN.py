# HIV Research
# General Regression Neural Network prototype
# Creadted May 29, 2014
# Updated June 1, 2014

import numpy as np
import math

# ================================================================================
# calculate weighted euclidean distance
def SqrWeightedEucludeanDist(x1, x2, sigma):
  dist = sum(((x1-x2)/sigma)**2)
#  print dist
  return dist

# ================================================================================
# predict y-hat
def predictYHat(idx, X, y, sigma):

  num = 0
  denom = 0
   
  # find the numerator and denominator
  for i in range(X.shape[0]):
    dist = SqrWeightedEucludeanDist(X[idx],X[i],sigma)
    num += y[i]*math.exp(-dist)
    denom += math.exp(-dist)

  # return y-hat
  return (num/denom)

# ================================================================================
# cost function (used to determine the cost of using a specific sigma value
def cost(yHat, y):
  samples = y.shape[0]
  cost = sum((yHat-y)**2)/samples
  return cost

# ================================================================================
# main function
def main():
  
  # data
  X = np.array([[1.0,2.0,3.0,4.0,5.0],
                [1.0,1.0,1.0,1.0,1.0],
                [2.0,3.0,1.5,2.0,1.9],
                [1.5,2.5,2.0,2.0,2.5]])
  
  # y values
  y = np.array([6.0,1.3,2.4,1.8])
  
  # array to hold predicted values
  yHat = np.zeros(y.shape[0])
  
  sigmas = np.arange(.001,1,.005) # values of sigma to test
  costs = np.zeros(sigmas.shape[0]) # find costs of using each sigma
  
  # best and worst set of predicted values
  # determined by cost function
  bestYHat = np.zeros(y.shape[0])
  worstYHat = np.zeros(y.shape[0])
  
  # best and worst starting values
  bestCost = 100000
  worstCost = 0
  
  # for each sigma value to test
  for i in range(sigmas.shape[0]):
    # for each sample
    for idx in range(y.shape[0]):
      # find predicted value
      yHat[idx] = predictYHat(idx, X, y, sigmas[i])
  
    print y
    print np.around(yHat, decimals=5)
  
    # determine cost of using sigma
    costs[i] = np.around(cost(yHat, y), decimals=5)
  
    # if cost is less than best cost, store predicted values
    if (costs[i] < bestCost):
      for k in range(yHat.shape[0]):
        bestYHat[k] = yHat[k]
      bestCost = costs[i]
      
    # if cost is greater than worst cost, store predicted values
    if (costs[i] >= worstCost):
      for k in range(yHat.shape[0]):
        worstYHat[k] = yHat[k]
      worstCost = costs[i]

    # print info
    print("sigma {}, cost {}, best {}, worst{}".format(sigmas[i], costs[i], bestCost, worstCost))
  
  # print best sigma value (makes lowest cost)
  print("\nBest sigma: {}".format(sigmas[np.argmin(costs)]))
  
  # print best and worst predicted values
  print("\nBest y-hat   Worst y-hat")
  for i in range(y.shape[0]):
    print("{}, {}".format(bestYHat[i], worstYHat[i]))

# ================================================================================
main()
# ================================================================================




























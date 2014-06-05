import numpy as np
import math

class GRNN:
  
  def __init__(self):
    self.bestSigma = None
  
  # find sigma???
  def fit(self, X, y):
    
    # array to hold predicted values
    yHat = np.zeros(y.shape[0])
    
    sigma = np.arange(.001,1,.005) # values of sigma to test
    costs = np.zeros(sigma.shape[0]) # find costs of using each sigma
    
    for i in range(sigma.shape[0]):
      for idx in range(y.shape[0]):
        yHat[idx] = self.predict(idx, X, y, sigma[i])
      # determine cost of using sigma
      costs[i] = np.around(self.cost(yHat, y), decimals=5)
    

    self.bestSigma = sigma[np.argmin(costs)]
    #print "Min cost : {}      best sigma : {}".format(min(costs), self.bestSigma)
    return 'GRNN'
  
  
  def predict(self, idx, X, y, sigma=None): # don't include sigma or y, use self.sigma
    if(sigma == None):
      sigma = self.bestSigma;
      #print "Changed sigma to {}".format(sigma)
    
    # useful variables
    num = 0
    denom = 0

    # find the numerator and denominator
    for i in range(X.shape[0]):
      dist = self.SqrWeightedEucludeanDist(X[idx],X[i],sigma)
      num += y[i]*math.exp(-dist)
      denom += math.exp(-dist)
    
    # return y-hat
    return (num/denom)
  
  # calculate weighted euclidean distance
  def SqrWeightedEucludeanDist(self, x1, x2, sigma):
    dist = sum(((x1-x2)/sigma)**2)
    return dist
  
  def cost(self, yHat, y):
    samples = y.shape[0]
    cost = sum((yHat-y)**2)/samples
    return cost


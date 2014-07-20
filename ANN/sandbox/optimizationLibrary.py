# Container reference for sandbox 
# testing Numba optimized functions
# ===
from numbapro import vectorize, cuda, jit
from numba import float32
import numpy as np
import math
import time

@vectorize(['float64(float64, float64)'], target='cpu') # default to 'cpu'
def add2(a, b):
    return a + b

@vectorize(['float64(float64)'], target='cpu') # default to 'cpu'
def cu_mult2(a):
    momentum = .1
    return a * momentum

# Create a ufunc
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def cuda_sumfunc(a, b):
    return a + b

# Create a ufunc
@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'], target='cpu')
def opt_sum(a, b):
    #return cuda_sumfunc(a, b);
    return a + b

@cuda.jit('float32(float32)', device=True, inline=True)
def cu_sigmoidfunc(a):
    return 1. / (1. + math.exp(-a));

@vectorize(['float32(float32)',
            'float64(float64)'], target='cpu') # defaults to 'cpu'
def opt_sigmoid(a):
    """ Logistic sigmoid function. """
    return 1. / (1. + math.exp(-a));
    #return cu_sigmoidfunc(a);

@cuda.jit('float32(float32, float32)', device=True, inline=True)
def cudaBPCalcfunc(a, b):
    return a * (1 - a) * b

# Create a ufunc
@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'], target='cpu')
def backPropagationFormula(a, b):
    #return cudaBPCalcfunc(a, b)
    return a * (1 - a) * b

@cuda.jit('float32(float32, float32)', device=True, inline=True)
def cu_gradientfunc(a, b):
    weightdecay = 0.0
    return a - weightdecay * b;

# Create a ufunc
@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'], target='cpu')
def opt_gradient(a, b):
    weightdecay = 0.0
    return a - weightdecay * b;
    #return cu_gradientfunc(a, b);
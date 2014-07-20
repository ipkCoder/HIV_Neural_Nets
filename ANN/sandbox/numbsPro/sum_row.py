from numbapro import vectorize
from numpy import arange

@vectorize(['float32(float32, float32)'], target='gpu') # default to 'cpu'
def add2(a, b):
    return a + b

X = arange(10, dtype='float32')
Y = X * 2
print add2(X, Y)
print add2.reduce(X)
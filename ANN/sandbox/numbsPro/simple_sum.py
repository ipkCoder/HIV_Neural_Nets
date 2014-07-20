import numpy
from numbapro import vectorize

# Create a ufunc
@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'], target='gpu')
def sum(a, b):
    return a + b

# Use the ufunc
dtype = numpy.float32
a = numpy.arange(10,dtype=dtype)
b = numpy.arange(10,dtype=dtype)
result = sum(a, b)      # call the ufunc

print("a = %s" % a)
print("b = %s" % b)
print("sum = %s" % result)
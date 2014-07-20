from numbapro.cudalib import curand
from numpy import empty
prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
rand = empty(10)
prng.uniform(rand)
print rand
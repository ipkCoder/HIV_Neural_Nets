import math
from numbapro import vectorize, cuda
import numpy as np

@vectorize(['float32(float32, float32, float32)',
            'float64(float64, float64, float64)'],
           target='gpu')
def cu_discriminant(a, b, c):
    return math.sqrt(b ** 2 - 4 * a * c)

N = 1e+4
dtype = np.float32

# prepare the input
A = np.array(np.random.sample(N), dtype=dtype)
B = np.array(np.random.sample(N) + 10, dtype=dtype)
C = np.array(np.random.sample(N), dtype=dtype)

D = cu_discriminant(A, B, C)

print(D)  # print result
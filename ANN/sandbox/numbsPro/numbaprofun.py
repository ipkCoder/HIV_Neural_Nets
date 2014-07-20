# Get all the imports we need
import numba.cuda
from numbapro import vectorize
import numpy as np
import math
import timeit

my_gpu = numba.cuda.get_current_device()
print "Running on GPU:", my_gpu.name
cores_per_capability = {
    1: 8,
    2: 32,
    3: 192,
}
cc = my_gpu.compute_capability
print "Compute capability: ", "%d.%d" % cc, "(Numba requires >= 2.0)"
majorcc = cc[0]
print "Number of streaming multiprocessor:", my_gpu.MULTIPROCESSOR_COUNT
cores_per_multiprocessor = cores_per_capability[majorcc]
print "Number of cores per mutliprocessor:", cores_per_multiprocessor
total_cores = cores_per_multiprocessor * my_gpu.MULTIPROCESSOR_COUNT
print "Number of cores on GPU:", total_cores

# CPU version
@numba.vectorize(['float32(float32, float32)',
                  'float64(float64, float64)'], target='cpu')
def cpu_sincos(x, y):
    return math.sin(x) * math.cos(y)

# CUDA version
@vectorize(['float32(float32, float32)',
                     'float64(float64, float64)'], target='gpu')
def gpu_sincos(x, y):
    return math.sin(x) * math.cos(y)

# Generate data
n = 1000000
x = np.linspace(0, np.pi, n)
y = np.linspace(0, np.pi, n)

# Check result
np_ans = np.sin(x) * np.cos(y)
nb_cpu_ans = cpu_sincos(x, y)
nb_gpu_ans = gpu_sincos(x, y)

print "CPU vectorize correct: ", np.allclose(nb_cpu_ans, np_ans)
print "GPU vectorize correct: ", np.allclose(nb_gpu_ans, np_ans)

@numba.vectorize(['float32(float32, float32, float32, float32)'])
def cpu_powers(p, q, r, s):
    return math.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)

@vectorize(['float32(float32, float32, float32, float32)'], target='gpu')
def gpu_powers(p, q, r, s):
    return math.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)

# Generate data
n = 5000000
p = np.random.random(n).astype(np.float32)
q = np.random.random(n).astype(np.float32)
r = np.random.random(n).astype(np.float32)
s = np.random.random(n).astype(np.float32)

# Check results
np_ans = np.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)
cpu_ans = cpu_powers(p, q, r, s)
gpu_ans = gpu_powers(p, q, r, s)
print "CPU vectorize correct", np.allclose(np_ans, cpu_ans)
print "GPU vectorize correct", np.allclose(np_ans, gpu_ans)
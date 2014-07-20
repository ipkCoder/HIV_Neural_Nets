import numbapro
import numba, numba.cuda
import numpy as np
import math
from qsarHelpers import *
from numba import cuda

print "numbapro", numbapro.__version__

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

@numba.cuda.jit("void(float32[:], float32[:], float32[:])")
def vadd(arr_a, arr_b, arr_out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x    # number of threads per block
    i = tx + bx * bw
    if i >= arr_out.size:
        return
    arr_out[i] = arr_a[i] + arr_b[i];

tx = cuda.threadIdx.x   # thread label (along x dimension)
bx = cuda.blockIdx.x    # block label (along x dimension)
bw = cuda.blockDim.x    # number of threads in each block (along x dimension)
i = tx + bx * bw        # flattened linear address for each thread

i = cuda.grid(1)

if i >= arr_out.size:
    print "foo";

arr_out[i] = arr_a[i] + arr_b[i]

n = 100
a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)
c = np.empty_like(a)                 # Must prepare the output array to hold the result

thread_ct = my_gpu.WARP_SIZE
block_ct = int(math.ceil(float(n) / thread_ct))

print "Threads per block:", thread_ct
print "Block per grid:", block_ct

vadd[block_ct, thread_ct](a, b, c)    # Last argument is the output array in this case
print c

#===
# Use the builtin matrix_multiply in NumPy for CPU test
import numpy.core.umath_tests as ut

# Example 3!! - Vectorizing array matrices (not-working cuLaunchKernel error code 701 thrown
# review for optimizing QSAR-ANN dataset second round
@numbapro.guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'],
                      '(m, n),(n, p)->(m, p)', target='gpu')
def batch_matrix_mult(a, b, c):
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tmp = 0
            for n in range(a.shape[1]):
                 tmp += a[i, n] * b[n, j]
            c[i, j] = tmp

def check_pure_compute_time(da, db, dc):
    batch_matrix_mult(da, db, out=dc)
    numba.cuda.synchronize()   # ensure the call has completed

n = 4000000
dim = 2
a = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)
b = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)

dc = numba.cuda.device_array_like(a)
da = numba.cuda.to_device(a)
db = numba.cuda.to_device(b)

print 'NumPy time'
with Timer() as t:
    ut.matrix_multiply(a, b)
print("Time : {:.03f} sec.".format(t.interval));

#print 'Numba GPU time'
#with Timer() as t:
#    batch_matrix_mult(a, b)
#print("Time : {:.03f} sec.".format(t.interval));

print 'Pure Numba GPU time'
with Timer() as t:
    check_pure_compute_time(da, db, dc)
print("Time : {:.03f} sec.".format(t.interval));
del da, db, dc


#=====
# CPU version
#@numba.vectorize(['float32(float32, float32)',
#                  'float64(float64, float64)'], target='cpu')
#def cpu_sincos(x, y):
#    return math.sin(x) * math.cos(y)
#
## CUDA version
#@numbapro.vectorize(['float32(float32, float32)',
#                     'float64(float64, float64)'], target='gpu')
#def gpu_sincos(x, y):
#    return math.sin(x) * math.cos(y)
#
## Generate data
#n = 1000000
#x = np.linspace(0, np.pi, n)
#y = np.linspace(0, np.pi, n)
#
## Check result
#np_ans = np.sin(x) * np.cos(y)
#nb_cpu_ans = cpu_sincos(x, y)
#nb_gpu_ans = gpu_sincos(x, y)
#
#print "CPU vectorize correct: ", np.allclose(nb_cpu_ans, np_ans)
#print "GPU vectorize correct: ", np.allclose(nb_gpu_ans, np_ans)
#
#print "NumPy"
#with Timer() as t:
#    np.sin(x) * np.cos(y)
#print("Time : {:.03f} sec.".format(t.interval));
#
#print "CPU vectorize"
#with Timer() as t:
#    cpu_sincos(x, y)
#print("Time : {:.03f} sec.".format(t.interval));
#
#print "GPU vectorize"
#with Timer() as t:
#    gpu_sincos(x, y)
#print("Time : {:.03f} sec.".format(t.interval));
#
## Optional cleanup 
#del x, y

#====
# Example 2: polynomial
#@numba.vectorize(['float32(float32, float32, float32, float32)'])
#def cpu_powers(p, q, r, s):
#    return math.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)

#@numbapro.vectorize(['float32(float32, float32, float32, float32)'], target='gpu')
#def gpu_powers(p, q, r, s):
#    return math.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)
## Generate data
#n = 5000000
#p = np.random.random(n).astype(np.float32)
#q = np.random.random(n).astype(np.float32)
#r = np.random.random(n).astype(np.float32)
#s = np.random.random(n).astype(np.float32)
#
## Check results
#np_ans = np.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)
#cpu_ans = cpu_powers(p, q, r, s)
#gpu_ans = gpu_powers(p, q, r, s)
#print "CPU vectorize correct", np.allclose(np_ans, cpu_ans)
#print "GPU vectorize correct", np.allclose(np_ans, gpu_ans)
#
#
#print "NumPy"
#with Timer() as t:
#    np.sqrt(p ** 2 + q ** 3 + r ** 4 + s ** 5)
#print("Time ex2 : {:.03f} sec.".format(t.interval));
#
#print "CPU vectorize"
#with Timer() as t:
#    cpu_powers(p, q, r, s)
#print("Time ex2 : {:.03f} sec.".format(t.interval));
#
#print "GPU vectorize"
#with Timer() as t:
#    gpu_powers(p, q, r, s)
#print("Time ex2 : {:.03f} sec.".format(t.interval));

# #Optional cleanup 
#del p, q, r, s
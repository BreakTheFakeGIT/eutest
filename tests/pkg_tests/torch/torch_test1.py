import numpy as np
import torch as pt
from timeit import default_timer as timer

#Basic Info
print(f"Version: {pt.__version__}, GPU: {pt.cuda.is_available()}, NUM_GPU: {pt.cuda.device_count()}")

#Set device to GPU if available
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(pt.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(pt.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(pt.cuda.memory_reserved(0)/1024**3,1), 'GB')

#Create a random tensor
x = pt.rand(5, 3)
print(x)

#func1 will run on the CPU
def func1(a):
    a+= 1

#func2 will run on the GPU
def func2(a):
    a+= 2

if __name__=="__main__":
    n1 = 300000000
    a1 = np.ones(n1, dtype = np.float64)

    # had to make this array much smaller than
    # the others due to slow loop processing on the GPU
    n2 = 300000000
    a2 = pt.ones(n2,dtype=pt.float64)

    start = timer()
    func1(a1)
    print("Timing with CPU:numpy", timer()-start)

    start = timer()
    func2(a2) 
    #wait for all calcs on the GPU to complete
    pt.cuda.synchronize()
    print("Timing with GPU:pytorch", timer()-start)
    print()

    print("a1 = ",a1)
    print("a2 = ",a2)
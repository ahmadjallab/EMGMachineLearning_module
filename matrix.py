#matrix 
# dim: 2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
'''
this for the matrix linear algebra examples
for computing the matrix and visualize  it

    integers: 2^(8) = 256 2^(16) = 65536 2^(32) = 4294967296 2^(64) = 18446744073709551616

    np.int8: 1 byte (8 bits)
    np.int16: 2 bytes (16 bits)
    np.int32: 4 bytes (32 bits)
    np.int64: 8 bytes (64 bits)
    unsigned integers:

    np.uint8: 1 byte (8 bits)
    np.uint16: 2 bytes (16 bits)
    np.uint32: 4 bytes (32 bits)
    np.uint64: 8 bytes (64 bits)
    floating-point numbers:

    np.float16: 2 bytes (16 bits)
    np.float32: 4 bytes (32 bits)
    np.float64: 8 bytes (64 bits)
    complex numbers:

    np.complex64: 8 bytes (64 bits, 32 bits for real, 32 bits for imaginary)
    np.complex128: 16 bytes (128 bits, 64 bits for real, 64 bits for imaginary)
    bool:

    np.bool: 1 byte (8 bits)
    object:

    np.object: varies (references to Python objects, size depends on the object)
    string:

    np.str_, np.unicode_: size varies based on the length of the string and encoding
'''

two_d_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=np.complex128)

# print(two_d_matrix)
# print (two_d_matrix.shape)
#how much tack from memory 
# print (two_d_matrix.nbytes)

CTD =two_d_matrix.nbytes*8 
# print (pow(2,CTD))
# convert from binary to decimal
# print (two_d_matrix.itemsize)


# load the data from json file

with open('data.json') as f:
  data = json.load(f) 
  
print (type(data))   
data = np.dtype(data,dtype=np.int32)


print ( data)
#print  with tabulate
#print(tabulate(data, tablefmt="fancy_grid"))


# Create a sample matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Set print options for NumPy
np.set_printoptions(precision=2, suppress=True)

# Print the matrix
print("Matrix:")
#print(tabulate(, tablefmt="fancy_grid"))

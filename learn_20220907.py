#stage 1
from pickletools import optimize
from scipy import constants
import scipy
print(constants.acre)
print(scipy.__version__)
#stage 2
from scipy.optimize import root
from math import cos
def eqn1(x):
    return x + cos(x)
myroot = root(eqn1,0)
print(myroot.x)
print(myroot)
#stage 3
from scipy.optimize import minimize
def eqn2(x):
    return x**2 + x + 2
mymin = minimize(eqn2, 0, method='BFGS')
print(mymin)
#stage 4
import numpy as np
from scipy.sparse import csr_matrix
arr1 = np.array([0,0,0,0,0,1,1,0,2])
print(csr_matrix(arr1))
print(csr_matrix(arr1).data)
arr2 = np.array([[0,0,0],[0,0,1],[1,0,2]])
arr3 = arr2
print(csr_matrix(arr2).count_nonzero())
# print(csr_matrix(arr2).eliminate_zeros())
mat1 = csr_matrix(arr2)
mat1.eliminate_zeros()
print(mat1)
mat2 = csr_matrix(arr3)
mat2.sum_duplicates()
print(mat2)
#stage 5

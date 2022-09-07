#stage 1
from pickletools import optimize
from turtle import color
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
from scipy.sparse.csgraph import connected_components,dijkstra,floyd_warshall
arr4 = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
newarr1 = csr_matrix(arr4)
print(connected_components(newarr1))
print(dijkstra(newarr1,return_predecessors=True,indices=0))
print(floyd_warshall(newarr1,return_predecessors=True))
#stage 6
from scipy.sparse.csgraph import bellman_ford
arr5 = np.array([
    [0,-1,2],
    [1,0,0],
    [2,0,0]
])
newarr = csr_matrix(arr5)
print(bellman_ford(newarr,return_predecessors=True,indices=0))
#stage 7
from scipy.sparse.csgraph import depth_first_order
arr = np.array([ 
    [0,1,0,1],
    [1,1,1,1],
    [2,1,1,0],
    [0,1,0,1]
])
newarr = csr_matrix(arr)
print(depth_first_order(newarr,1))
#stage 8
from scipy.sparse.csgraph import breadth_first_order
print(breadth_first_order(newarr,1))
#stage 9
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
points = np.array([ 
    [2,4],
    [3,4],
    [3,0],
    [2,2],
    [4,1]
])
simplices = Delaunay(points).simplices
plt.triplot(points[:,0],points[:,1],simplices)
plt.scatter(points[:,0],points[:,1],color='r')
plt.show()
#stage 10
from scipy.spatial import ConvexHull
points = np.array([ 
    [2,4],
    [3,4],
    [3,0],
    [2,2],
    [4,1],
    [1,2],
    [5,0],
    [3,1],
    [1,2],
    [0,2]
])
hull = ConvexHull(points)
hull_points = hull.simplices
plt.scatter(points[:,0],points[:,1])
for simplex in hull_points:
    plt.plot(points[simplex,0],points[simplex,1],'k-')
plt.show()
#stage 11
from scipy.spatial import KDTree
points = [(1,-1),(2,3),(-2,3),(2,-3)]
kdtree = KDTree(points)
res = kdtree.query((1,1))
print(res)

import numpy as np
from numpy.linalg import inv, solve

A = np.array([
    [1, 2, 3],
    [3, 5, 6],
    [0, 8, 9] 
])

print("Transpozycja:")
print(A.transpose())

print(inv(A))

B = np.array([3, 2, 1])

print(solve(A, B))

print(A @ B)
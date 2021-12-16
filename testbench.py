import numpy as np

from SEXTANS import SEXTANS


# Test if 2 numpy arrays are equal to the 4th decimal place
def arr_equal(arr1, arr2):
    if np.array_equal(np.around(arr1, decimals=4), np.around(arr2, decimals=4)):
        return True
    else:
        return False


# Create a random matrix with given dimensions, percent of overall elements that are non-zero, and max/min values
def rand_sparse_arr(dims, percentNonZero, maxMin):
    sparse = np.random.rand(dims[0], dims[1])
    sparse = (np.random.rand(dims[0], dims[1]) * (maxMin[1] - maxMin[0]) + maxMin[0]) * (sparse < percentNonZero)
    return sparse


# Test a variety of matrix dimensions
for M in range(64, 600, 101):
    for K in range(64, 600, 101):
        for N in range(8, 16, 3):
            s = SEXTANS(M, 8, 100, 100, 8, 0)
            A = rand_sparse_arr([M, K], 0.1, [-50, 50])
            B = rand_sparse_arr([K, N], .9, [-50, 50])

            test = s.dot_multiply(A, B)
            true = np.dot(A, B)

            if not arr_equal(test, true):
                print(M, K, N, 'failure')
            else:
                print(M, K, N, 'success')

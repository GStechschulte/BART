import os
import sys
import timeit

from copy import copy

import numba
import numpy as np

from numba import njit

# Access modules from parent directory since `pymc_bart` is not an installable package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from pymc_bart.leaf_ops import fast_mean_cython


@njit
def fast_mean(ari):
    """Use Numba to speed up the computation of the mean."""

    if ari.ndim == 1:
        count = ari.shape[0]
        suma = 0
        for i in range(count):
            suma += ari[i]
        return suma / count
    else:
        res = np.zeros(ari.shape[0])
        count = ari.shape[1]
        for j in range(ari.shape[0]):
            for i in range(count):
                res[j] += ari[j, i]
        return res / count


def test_fast_mean_cython():
    # Define the range of sizes and number of arrays to generate
    min_size = 5
    max_size = 20
    num_arrays = 10
    sizes = [i * 10 for i in range(1, 11)]

    # Generate random arrays of different sizes
    arrays = []
    for size in sizes:
        arr = np.random.rand(size)
        arrays.append(arr)

    # Define the input sizes to test
    input_sizes = [1, 2, 3, 4]

    # Measure the execution time of the Cython function for each input size
    for arr in arrays:
        cython_time = timeit.timeit(lambda: fast_mean_cython(arr), number=10000)
        print(
            f"Cython: fast_mean_cython({len(arr)}) = {fast_mean_cython(arr)}, time = {cython_time:.6f} seconds"
        )

    # Measure the execution time of the Python function for each input size
    for arr in arrays:
        python_time = timeit.timeit(lambda: fast_mean(arr), number=10000)
        print(
            f"Python: fast_mean({len(arr)}) = {fast_mean(arr)}, time = {python_time:.6f} seconds"
        )

if __name__ == "__main__":
    test_fast_mean_cython()
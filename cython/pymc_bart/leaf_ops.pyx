import cython
import numpy as np
cimport numpy as cnp


cpdef tuple draw_leaf_value_cython(
    cnp.ndarray[cnp.float64_t] y_mu_pred,
    cnp.ndarray[cnp.float64_t] x_mu,
    int m,
    cnp.ndarray[cnp.float64_t] norm,
    int shape,
    str response
    ):
    """Draw Gaussian distributed leaf values.
    """
    cdef bint has_linear_params = False
    cdef cnp.ndarray[cnp.float64_t] mu_mean = np.empty(shape)
    cdef cnp.ndarray[cnp.float64_t] linear_params = None

    if y_mu_pred == 0:
        return np.zeros(shape), None
    
    if y_mu_pred.size == 1:
        mu_mean[:] = y_mu_pred[0] / m + norm[:]
    elif y_mu_pred.size < 3 or response == "constant":
        mu_mean = fast_mean_cython(y_mu_pred) / m + norm
    
    return mu_mean, linear_params


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double fast_mean_cython(double[:] ari_view):
    """Use Cython to speed up the computation of the mean.

    Parameters
    ----------
    ari_view : memoryview
        A memoryview of the array to compute the mean of. `double[:]` indicates
        a 1D typed memoryview of doubles
    
    Returns
    -------
    double
        The mean of the array.
    """
    # `Py_ssize_t` is the proper C type for Python array indices
    cdef Py_ssize_t count = ari_view.shape[0]
    cdef Py_ssize_t n = ari_view.shape[1]
    cdef Py_ssize_t i, j
    
    cdef double suma = 0.0

    cdef double[:] res = np.zeros(count, dtype=np.float64)

    if ari_view.ndim == 1:
        for i in range(count):
            suma += ari_view[i]
        return suma / count
    else:
        for j in range(count):
            for i in range(n):
                # Access using stride information
                res[j] += ari_view[j * n + i]
        # Use memoryview's mean method
        return res.mean()


# TODO: Cythonize and use memory views
# def fast_linear_fit(
#     x: npt.NDArray[np.float_],
#     y: npt.NDArray[np.float_],
#     m: int,
#     norm: npt.NDArray[np.float_],
# ) -> Tuple[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]:
#     n = len(x)
#     y = y / m + np.expand_dims(norm, axis=1)

#     xbar = np.sum(x) / n
#     ybar = np.sum(y, axis=1) / n

#     x_diff = x - xbar
#     y_diff = y - np.expand_dims(ybar, axis=1)

#     x_var = np.dot(x_diff, x_diff.T)

#     if x_var == 0:
#         b = np.zeros(y.shape[0])
#     else:
#         b = np.dot(x_diff, y_diff.T) / x_var

#     a = ybar - b * xbar

#     y_fit = np.expand_dims(a, axis=1) + np.expand_dims(b, axis=1) * x
#     return y_fit.T, [a, b]
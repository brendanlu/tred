"""Various utility functions"""


import numpy as np
import os
from scipy.fft import dct, idct


def generate_DCTii_M_transform_pair(t):
    """Wrapper around scipy fft to generate functions to perform $\times_3 M$ operations
    on order-3 tensors, where $M$ is the (scaled) Discrete Cosine Transform.

    As introduced by Kilmer et al. (2021) and applied by Mor et al. (2022)

    For now, this is the default transform; this may change later.

    Parameters
    ----------
        t : int
            The length of the transform

    Returns
    -------
        fun_m : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies DCT-II to
            each of the tubal fibres. This preserves the dimensions of the tensor.

        inv_m : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `fun_m`)
    """

    def M(X):
        assert X.shape[-1] == t, f"Expecting last input dimension to be {t}"
        return dct(X, type=2, n=t, axis=-1, norm="ortho", workers=2 * os.cpu_count())

    def Minv(X):
        assert X.shape[-1] == t, f"Expecting last input dimension to be {t}"
        return idct(X, type=2, n=t, axis=-1, norm="ortho", workers=2 * os.cpu_count())

    return M, Minv


def _singular_vals_mat_to_tensor(mat, n, p, t):
    """Decompress $k \times t$ matrix of singular values into $\hat{S}$ from literature.

    Parameters
    ----------
        mat : ArrayLike, shape: (k, t_)
            $k \times t_$ matrix representation, the function checks if `t_ == t`

        n : int
            first dimension of output tensor

        p : int
            second dimension of output tensor

        t : int
            third dimensions of output tensor

    Returns
    -------
        hatS : ndarray, shape: (n, p, t)
            $n \times p \times t$ tensor representation
    """
    k = min(n, p)
    assert k == mat.shape[0] and t == mat.shape[1], "Ensure conforming dimensions"

    hatS = np.zeros((n, p, t))
    for i in range(t):
        hatS[:k, :k, i] = np.diag(mat[:, i])

    return hatS


def _singular_vals_tensor_to_mat(tensor):
    """Compresses $\hat{S}$, a $n \times p \times t$ f-diagonal tensor of singular values
    into a $k \times t$ matrix. Each row contains the $k$ singular values from that
    longitudinal slice.

    Reverses _singular_vals_mat_to_tensor

    Parameters
    ----------
        tensor : ArrayLike, shape: (n, p, t)
            $n \times p \times t$ tensor representation

    Returns
    -------
        mat : ndarray, shape: (min(n, p), t)
            $k \times t$ matrix representation of singular values, where
            $k = \min{(n,p)}$
    """
    return np.diagonal(tensor, axis1=0, axis2=1).transpose()

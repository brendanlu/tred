"""Utilities for generating and defining M transforms in the m-product framework"""


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

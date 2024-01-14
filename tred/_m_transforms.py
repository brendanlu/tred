"""Utilities for generating and defining M transforms in the m-product framework"""


import os
from scipy.fft import dct, idct, dst, idst


def generate_dctii_m_transform_pair(t, *, norm="ortho"):
    """Wrapper around scipy fft to generate functions to perform $\times_3 M$ operations
    on order-3 tensors, where $M$ is the (scaled) Discrete Cosine Transform.

    Parameters
    ----------
        t : int
            The length of the transform

        norm : {“backward”, “ortho”, “forward”}, default="ortho"
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct

    Returns
    -------
        M : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies DCT-II to
            each of the tubal fibres. This preserves the dimensions of the tensor.

        Minv : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `fun_m`)

    References
    ----------
    The use of this transform with the m-product was introduced in:
    `Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.`
    """

    def M(X):
        assert X.shape[-1] == t, f"Expecting last input dimension to be {t}"
        return dct(X, type=2, n=t, axis=-1, norm=norm, workers=2 * os.cpu_count())

    def Minv(X):
        assert X.shape[-1] == t, f"Expecting last input dimension to be {t}"
        return idct(X, type=2, n=t, axis=-1, norm=norm, workers=2 * os.cpu_count())

    return M, Minv


def generate_dstii_m_transform_pair(t, *, norm="ortho"):
    """Wrapper around scipy fft to generate functions to perform $\times_r M$ operations
    on order-3 tensor, where $M$ is the (scaled) Discrete Sine Transform.

    Parameters
    ----------
        t : int
            The length of the transform

        norm : {“backward”, “ortho”, “forward”}, default="ortho"
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html#scipy.fft.dst

    Returns
    -------
        M : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies DFT via a FFT
            algorithm to each of the tubal fibres. This preserves the dimensions of the
            tensor.

        Minv : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `fun_m`)
    """

    def M(X):
        assert X.shape[-1] == t, f"Expecting last input dimension to be {t}"
        return dst(X, type=2, n=t, axis=-1, norm=norm, workers=2 * os.cpu_count())

    def Minv(X):
        assert X.shape[-1] == t, f"Expecting last input dimension to be {t}"
        return idst(X, type=2, n=t, axis=-1, norm=norm, workers=2 * os.cpu_count())

    return M, Minv

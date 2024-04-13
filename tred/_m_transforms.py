"""Utilities for generating and defining M transforms in the m-product framework"""

import os
import sys

import numpy.linalg as linalg
from scipy.fft import dct, idct, dst, idst


# the transform generator functions below all work for:
#   1. 3D array (tensor): by applying the transform along the tubal fibres
#   2. 2D array (matrices): by applying the transform along the rows
#   3. 1D array (vector): by applying the transform on the vector
# i.e., along the -1 axis of order-2 and order-3 arrays
#
# we do not need extensive parameter validation however, as the M and Minv Callables
# returned by the generator functions are meant to be passed into classes and functions
# in the tred package - which do some of their own parameter validation
SUPPORTED_TRANSFORM_DIMENSIONS = (1, 2, 3)


def _assert_t_and_order(X_input, t_expected):
    assert (
        len(X_input.shape) in SUPPORTED_TRANSFORM_DIMENSIONS
    ), "Expecting 1D (vector), 2D (matrix) or 3D (tensor) input"
    assert (
        X_input.shape[-1] == t_expected
    ), f"Expecting last input dimension to be {t_expected}"


def generate_default_m_transform_pair(t):
    """Wrapper for a function that generates functions M and Minv given the length of
    a tubal fibre. Allows tred user to retrive the tubal transform functions for direct
    computations using the tensor-tensor m-product.

    The algorithms in the tred library do not require the user to explicitly specify a
    tensor m-transform. In those cases, they revert to the default specified in this
    function here.

    We may choose to choose a different default family of mappings later.

    Parameters
    ----------
        t : int
            The length of the transform (length of the tubal fibers)

    Returns
    -------
        M : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies a tensor
            transform to each of the tubal fibres. This preserves the dimensions of
            the tensor.

        Minv : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `fun_m`)

    References
    ----------
    The use of this transform with the m-product was introduced in:
    `Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.`
    """
    return generate_dctii_m_transform_pair(t)


def generate_transform_pair_from_matrix(M_mat, Minv_mat=None, *, inplace=False):
    """Generate a pair of functions to apply a matrix, and its inverse, to the tubal
    fibres of an order-3 tensor. See Kilmer et al. (20211).

    Parameters
    ----------
        M_mat : ArrayLike
            Square matrix

        Minv_mat : ArrayLike or None, default=None
            Square matrix, the inverse of M_mat. If not specified, this function will
            numerically evaluate the inverse of `M_mat`

        inplace : bool, default=False
            Control whether or not the generated functions modify the input tensor
            in-place, or return a copy with the m-transform applied

            THIS ARGUMENT DOES NOT CURRENTLY ALTER ANY BEHAVIOUR FOR THIS FUNCTION, BUT
            IS A PLACEHOLDER FOR FUTURE DEVELOPMENT.

    Returns
    -------
        M : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies `M_mat`
            to each of the tubal fibres. This preserves the dimensions of the tensor.

        Minv : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `M`)

    References
    ----------
    `Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data. Proceedings
    of the National Academy of Sciences, 118(28), p.e2015851118.`
    """
    assert len(M_mat.shape) == 2, "Expecting matrix (order-2 array) input"
    assert M_mat.shape[0] == M_mat.shape[1], "Expecting square matrix input"
    t_ = M_mat.shape[0]

    # numerically evaluate inverse matrix if not explicitly specified
    if Minv_mat is None:
        # be pedantic and check for invertibility
        # https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        # NOTE: revisit later, is the following more robust?:
        # https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy
        if linalg.cond(M_mat) < 1 / sys.float_info.epsilon:
            Minv_mat = linalg.inv(M_mat)
        else:
            raise ValueError(
                "Input matrix must be invertible, but appears singular, or close to singular"
            )
    else:
        assert (
            Minv_mat.shape == M_mat.shape
        ), "Ensure the shapes of matrix M and its inverse are the same"

    # a quick way of applying M along the tubal fibres of X, using np broadcasting:
    # transpose X into a n x t x p vertical stack of t x p matrices. matrix
    # multiplication is broadcast vertically, whereby M multiplies each of the t x p
    # matrices, effectively applying the transform to the columns, which were the original
    # tubal fibres. then transpose back to original
    def M(X):
        _assert_t_and_order(X, t_)
        if len(X.shape) == 3:
            return (M_mat @ X.transpose(0, 2, 1)).transpose(0, 2, 1)
        elif len(X.shape) == 2:
            return (M_mat @ X.T).T
        else:  # len(X.shape == 1)
            return M_mat @ X

    def Minv(X):
        _assert_t_and_order(X, t_)
        if len(X.shape) == 3:
            return (Minv_mat @ X.transpose(0, 2, 1)).transpose(0, 2, 1)
        elif len(X.shape) == 2:
            return (Minv_mat @ X.T).T
        else:  # len(X.shape == 1)
            return Minv_mat @ X

    return M, Minv


def generate_dctii_m_transform_pair(t, *, inplace=False, norm="ortho"):
    """Wrapper around scipy fft to generate functions to perform $\times_3 M$ operations
    on order-3 tensors, where $M$ is the (scaled) Discrete Cosine Transform.

    Parameters
    ----------
        t : int
            The length of the transform

        inplace : bool, default=False
            Control whether or not the generated functions modify the input tensor
            in-place, or return a copy with the m-transform applied

        norm : {“backward”, “ortho”, “forward”}, default="ortho"
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct

    Returns
    -------
        M : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies DCT-II to
            each of the tubal fibres. This preserves the dimensions of the tensor.

        Minv : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `M`)

    References
    ----------
    The use of this transform with the m-product was introduced in:
    `Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.`
    """

    def M(X):
        _assert_t_and_order(X, t)
        return dct(
            X,
            type=2,
            n=t,
            axis=-1,
            norm=norm,
            workers=2 * os.cpu_count(),
            overwrite_x=inplace,
        )

    def Minv(X):
        _assert_t_and_order(X, t)
        return idct(
            X,
            type=2,
            n=t,
            axis=-1,
            norm=norm,
            workers=2 * os.cpu_count(),
            overwrite_x=inplace,
        )

    return M, Minv


def generate_dstii_m_transform_pair(t, *, inplace=False, norm="ortho"):
    """Wrapper around scipy fft to generate functions to perform $\times_r M$ operations
    on order-3 tensor, where $M$ is the (scaled) Discrete Sine Transform.

    Parameters
    ----------
        t : int
            The length of the transform

        inplace : bool, default=False
            Control whether or not the generated functions modify the input tensor
            in-place, or return a copy with the m-transform applied

        norm : {“backward”, “ortho”, “forward”}, default="ortho"
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html#scipy.fft.dst

    Returns
    -------
        M : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies DFT via a FFT
            algorithm to each of the tubal fibres. This preserves the dimensions of the
            tensor.

        Minv : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `M`)
    """

    def M(X):
        _assert_t_and_order(X, t)
        return dst(
            X,
            type=2,
            n=t,
            axis=-1,
            norm=norm,
            workers=2 * os.cpu_count(),
            overwrite_x=inplace,
        )

    def Minv(X):
        _assert_t_and_order(X, t)
        return idst(
            X,
            type=2,
            n=t,
            axis=-1,
            norm=norm,
            workers=2 * os.cpu_count(),
            overwrite_x=inplace,
        )

    return M, Minv

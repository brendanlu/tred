"""Utilities for defining various types of m-transforms
"""

import os
import sys

import numpy.linalg as linalg
from scipy.fft import dct, idct, dst, idst


# the transform generator functions below all work for:
#    - 3D arrays (tensors)  : by applying the transform along the tubal fibres
#    - 2D arrays (matrices) : by applying the transform along the rows
#    - 1D arrays (vectors)  : by applying the transform on the vector
# NOTE: just along the -1 axis of order 1,2,3 ndarrays
SUPPORTED_TRANSFORM_DIMENSIONS = (1, 2, 3)


def _assert_t_and_order(X_input, t_expected):
    assert (
        len(X_input.shape) in SUPPORTED_TRANSFORM_DIMENSIONS
    ), "Expecting 1D (vector), 2D (matrix) or 3D (tensor) input"
    assert (
        X_input.shape[-1] == t_expected
    ), f"Expecting last input dimension to be {t_expected}"


def generate_default_m_transform_pair(t):
    """Return the default `M, Minv` used by `tred` algorithms.

    We do not guarantee that the default m-transform used in `tred` will remain
    consistent between versions, as this may change depending on recent
    research and literature. We highly encourage `tred` users to explicitly
    define M and Minv in the calling state to 'future-proof' your code.

    Parameters
    ----------
    t : int
        The length of the tubal fibers of the target tensors, i.e. the size of
        its third dimension.

    Returns
    -------
    M : Callable[[ndarray], ndarray]
        A function which expects a ndarray input, and returns the m-transform.

    Minv : Callable[[ndarray], ndarray]
        A function implementing the inverse transform of `M`

    References
    ----------
    Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data.
    Proceedings of the National Academy of Sciences, 118(28), p.e2015851118.

    Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.
    """
    return generate_dctii_m_transform_pair(t)


def generate_transform_pair_from_matrix(M_mat, Minv_mat=None, *, inplace=False):
    """Return `M, Minv` as defined by an orthogonal matrix.

    Allows the user to specify any orthogonal matrix, and this function will
    infer `t`, and numerically compute the inverse, returning functions `M`
    and `Minv` which can be used for `tred` algorithms.

    Optionally, the user can also choose to specify the inverse explicitly.

    Parameters
    ----------
    M_mat : ndarray
        Orthogonal square matrix.

    Minv_mat : ndarray or None, default=None
        Square matrix, the inverse of M_mat. If not specified, this function
        will numerically evaluate the inverse of `M_mat`.

    inplace : bool, default=False
        *Placeholder for future development*

    Returns
    -------
    M : Callable[[ndarray], ndarray]
        A function which expects an order-3 tensor as input, and applies
        `M_mat` to each of the tubal fibres. This preserves the dimensions of
        the tensor.

    Minv : Callable[[ndarray], ndarray]
        A function implementing the inverse transform of `M`

    References
    ----------
    Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data.
    Proceedings of the National Academy of Sciences, 118(28), p.e2015851118.
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

    # a quick way of applying M along the tubal fibres of X, using numpy
    # broadcasting: transpose X into a (n x t x p) vertical stack of (t x p)
    # matrices. matrix multiplication is broadcast vertically, whereby M left
    # multiplies each of the t x p matrices, effectively applying the
    # transform to the columns, which were the original tubal fibres.
    # then just perform the reverse transposition to get the original
    # orientation of the tensor
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
    """Return `M, Minv` as defined by the Discrete Cosine Transform of
    length `t`.

    Bascially a wrapper around scipy.fft to generate functions to perform
    fourier transform based $\\times_3 M$ operations, as used by
    Mor et al. (2022)

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
    M : Callable[[ndarray], ndarray]
        A function which expects an order-3 tensor as input, and applies the
        DCT-II transorm to each of the tubal fibres. This preserves the
        dimensions of the tensor.

    Minv : Callable[[ndarray], ndarray]
        A function implementing the inverse transform of `M`

    References
    ----------
    Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.
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
    """Return `M, Minv` as defined by the Discrete Sine Transform of
    length `t`.

    Bascially a wrapper around scipy.fft to generate functions to perform
    fourier transform based $\\times_3 M$ operations.

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
    M : Callable[[ndarray], ndarray]
        A function which expects an order-3 tensor as input, and applies the
        DST-II transorm to each of the tubal fibres. This preserves the
        dimensions of the tensor.

    Minv : Callable[[ndarray], ndarray]
        A function implementing the inverse transform of `M`
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

"""Module with mathematical operations"""


import numpy as np

from ._utils import _singular_vals_tensor_to_mat


def _facewise_product(A, B):
    """Facewise product s.t. $C_{:,:,i} = A_{:,:,i} B_{:,:,i}$

    Parameters
    ----------
        A : ArrayLike of shape (a, b, d)
            $a \times b \times d$ tensor representation

        B : ArrayLike of shape (b, c, d)
            $b \times c \times d$ tensor representation

    Returns
    -------
        C : ndarray, shape: (a, c, d)
            facewise tensor product

    """
    # return np.einsum('mpi,pli->mli', A, B)
    # the following is a quicker version of the above using numpy broadcasting
    return np.matmul(A.transpose(2, 0, 1), B.transpose(2, 0, 1)).transpose(1, 2, 0)


def _m_product(A, B, M, Minv):
    """Kilmer et al. (2021) tensor m-product for order-3 tensors. See [1]

    NOTE: NOT USED so far in this package. We tend to keep variables in the transformed
    space, and therefore directly use the _facewise_product function.

    Parameters
    ----------
        A : ArrayLike of shape (a, b, d)
            $a \times b \times d$ tensor representation

        B : ArrayLike of shape (b, c, d)
            $b \times c \times d$ tensor representation

        M : Callable[[ArrayLike], ndarray]
            A function which, given some order-3 tensor, returns it under some $\times_3$
            invertible transformation.

        MInv : Callable[[ArrayLike], ndarray]
            The inverse transformation of M

    Returns
    -------
        m_product : ndarray, shape: (a, c, d)
            Tensor-tensor m-product as found in Kilmer et al. (2021)
    """
    assert (
        A.shape[1] == B.shape[0] and A.shape[2] == B.shape[2]
    ), "Non conforming dimensions"

    return Minv(_facewise_product(M(A), M(B)))


def _rank_q_truncation_zero_out(hatU, hatS, hatV, *, q=None, sigma_q=None):
    """Explicit rank-q truncation as in Mor et al. (2022). Truncate, via zeroing out, the
    input arrays passed in by reference.

    This will typically be called in the m-transformed ('hat') space.

    Parameters
    ----------
        hatU : ArrayLike of shape (n, k, t)
            tensor U returned by tSVDM

        hatS : ArrayLike of shape (k, k, t) or (k, t)
            singular values tensor, or matrix representation, from tSVDM

        hatV : ArrayLike of shape (p, k, t)
            tensor V returned by tSVDM

        q : int
            Explicit rank ``q``, of the truncation

        sigma_q : float
            If ``sigma_q`` is set, then the ``q`` input parameter will be ignored. Saves
            computation time by passing in the q-th largest singular value, which
            may be cheap from the calling state if a sorted array of singular values has
            already been computed.

    Returns
    -------
        None
            Modifies inputs in-place
    """
    assert not (q is None and sigma_q is None), "Please specify either q or sigma_q"

    # assume hatS in matrix form
    if len(hatS.shape) == 3:
        hatS = _singular_vals_tensor_to_mat(hatS)

    k, t = hatS.shape

    # determine the q-th largest singular value
    if sigma_q is None:
        sigma_q = np.partition(hatS.flatten(), -q)[-q]

    # compute a representation of rho, as notated in Mor et al. (2022)
    # each diagonal of the frontal slice is already sorted in decreasing order
    # np.searchsorted is called on each column, reversed so that it is ascending,
    # where it returns the index i
    #   s.t. sigma_q <= ascending_col[i] (see numpy searchsorted docs)
    # which mean that we want to discard the i smallest values, and keep the other
    # k-i, which is the number we store in rho
    rho = np.apply_along_axis(
        lambda col: k - np.searchsorted(col[::-1], sigma_q, side="left"),
        axis=0,
        arr=hatS,
    )

    # perform multi-rank truncation, t should be modestly sized, so the for-loop
    # should be bearable
    # NOTE: there is probably some Numpy trick for this
    for i in range(t):
        hatU[:, rho[i] :, i] = 0
        hatS[rho[i] :, i] = 0
        hatV[:, rho[i] :, i] = 0

    return rho


def _mode0_unfold(tens):
    """As defined here https://jeankossaifi.com/blog/unfolding.html, there seem to be
    contradicting definitions and implementations using this same name.

    NOTE: NOT USED so far in this package. Probably redundant later if we adopt a more
    intuitive tensor package in Python.
    """
    # unfold the n x p x t tensor into a n x pt 2d array (matrix), where each frontal
    # slice sits 'next' each other. first transpose into a n-vertical-stack of t x p
    # matrices. looking downwards, with C-memory layout, ravel followed by reshaping into
    # a n x pt matrix gives the intended result; the unfolding has the same tensor
    # semantics as singular_values_ in the fit method
    return tens.transpose(0, 2, 1).reshape((tens.shape[0], -1), order="C")

"""Module with mathematical operations"""


from functools import reduce

import numpy as np

from ._m_transforms import generate_default_m_transform_pair
from ._utils import _singular_vals_tensor_to_mat

# np.einsum('mpi,pli->mli', tens1, tens2)
# the following is a quicker version of the above using numpy broadcasting
#
# NOTE: we have defined an anonymous functions for a binary facewise operation between
# two tensors defined by numpy ndarrays
#
# the facewise_product(*args) function works for any number of tensor inputs, and applies this
# operation cumulatively over the inputs
#
# however, the m_product(*args) function directly calls this binary lambda operation
# defined below, so that it can lazily reduce it over a python *generator*, thus saving
# us from having to store all the input tensors in the 'hat-space' (see Mor et al. 2022)
# in memory
_BINARY_FACEWISE = lambda tens1, tens2: (
    tens1.transpose(2, 0, 1) @ tens2.transpose(2, 0, 1)
).transpose(1, 2, 0)


def facewise_product(*tensors):
    """Compute cumulative facewise product s.t. $(C_{:,:,i} = A_{:,:,i} B_{:,:,i}$.

    Parameters
    ----------
        *tensors : ndarray
            Variable number of tensors, such that all adjacent input tensors have
            shape (a, b, d) and shape (b, c, d) respectively

    Returns
    -------
        C : ndarray, shape: (a, c, d)
            Facewise tensor product
    """
    # apply the lambda function cumulatively over the tensor inputs
    return reduce(_BINARY_FACEWISE, tensors)


def m_product(*tensors, **transforms):
    """Kilmer et al. (2021) tensor m-product for order-3 tensors.

    Parameters
    ----------
        *tensors : ndarray
            Variable number of tensors, such that all adjacent input tensors have
            shape (a, b, d) and shape (b, c, d) respectively

        M : Callable[[ArrayLike], ndarray] or None, default=None
            A function which, given some order-3 tensor, returns it under an orthogonal
            tubal transformation

        MInv : Callable[[ArrayLike], ndarray] or None, default=None
            A function implementing the inverse tubal transformation of M

    Returns
    -------
        m_product : ndarray, shape: (a, c, d)
            Tensor-tensor m-product as found in Kilmer et al. (2021)

    References
    ----------
    Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data. Proceedings
    of the National Academy of Sciences, 118(28), p.e2015851118.
    """
    # process the m-transform keyword inputs, and use default transforms if unspecified
    default_transforms = {"M": None, "Minv": None}
    transforms = {**default_transforms, **transforms}
    assert not (
        callable(transforms["M"]) ^ callable(transforms["Minv"])
    ), "If explicitly defined, both M and its inverse must be defined"

    if transforms["M"] is None:
        M, Minv = generate_default_m_transform_pair(tensors[0].shape[-1])
    else:
        M, Minv = transforms["M"], transforms["Minv"]

    # use a generator representing all the tensors under the m-transform ('hat space'),
    # and apply the binary facewise function over the generator; it will only lazily
    # evaluate the generator so we do not have to store all the transformed tensors in
    # memory at once
    #
    # then apply Minv at the end to get back into the 'original' (untransformed) tensor
    # space
    return Minv(reduce(_BINARY_FACEWISE, (M(tens) for tens in tensors)))


def _rank_q_truncation_zero_out(hatU, hatS, hatV, *, q=None, sigma_q=None):
    """In-place explicit rank-q truncation as introduced in Mor et al. (2022). Truncates
    tensors U, S, V from a tsvdm decomposition to achieve an explicit rank of q.

    Parameters
    ----------
        hatU : ndarray, shape (n, k, t)
            Tensor U from the tsvdm.

        hatS : ndarray, shape (k, k, t) or (k, t)
            Tensor S from the tsvdm, or represented in compact matrix form.

        hatV : ndarray, shape (p, k, t)
            Tensor V from the tSVDM

        q : int or None, default=None
            Target explicit rank for the truncation

        sigma_q : float or None, default=None
            The `q`-th largest singular value. This will not be checked, and assumed to
            be a valid singular value in the inputted decomposition.

            If `sigma_q` is set, then the `q` input parameter will be ignored. Saves
            re-computation of the `q`-th largest singular value.

    Returns
    -------
        rho : ndarray, shape (q,)
            The multi-rank which results from the choice of `q` (or `sigma_q`)

    References
    ----------
    Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.
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


def _mode_1_unfold(tens, view=False):
    """Return mode-1 unfolding copy, as defined in Kolda et al.

    References
    ----------
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications. SIAM
    review, 51(3), pp.455-500.
    """
    # unfold the n x p x t tensor into a n x pt 2d array (matrix), where each frontal
    # slice sits 'next' each other.
    #
    # NUMPY NOTES:
    # first transpose into a n-vertical-stack of t x p matrices. reshape is equivalent
    # to 'ravel'-ling first, before reshaping the vector (using the same memory format)
    # into the intended shape. in "C" memory format, the last indexes move the quickest,
    # so ravel moves along the p index, and then the t index, and then the n index.
    # when placing this into the matrix form, we get the intended result, where for the
    # same t and n, the p values sit next to each other; for the same n, all of its t
    # values sit together on the same row. then the matrix shape ensures that each row
    # has a distinct n
    #
    # the unfolding has the same tensor semantics as singular_values_ in the fit method
    if view:
        return tens.view().transpose(0, 2, 1).reshape((tens.shape[0], -1), order="C")
    else:
        return tens.copy().transpose(0, 2, 1).reshape((tens.shape[0], -1), order="C")


def _mode_2_unfold(tens, view=False):
    """Return mode-2 unfolding copy, as defined in Kolda et al.

    References
    ----------
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications. SIAM
    review, 51(3), pp.455-500.
    """
    if view:
        return tens.view().transpose(1, 2, 0).reshape((tens.shape[1], -1), order="C")
    else:
        return tens.copy().transpose(1, 2, 0).reshape((tens.shape[1], -1), order="C")


def _mode_3_unfold(tens, view=False):
    """Return mode-3 unfolding copy, as defined in Kolda et al.

    References
    ----------
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications. SIAM
    review, 51(3), pp.455-500.
    """
    if view:
        return tens.view().transpose(2, 1, 0).reshape((tens.shape[2], -1), order="C")
    else:
        return tens.copy().transpose(2, 1, 0).reshape((tens.shape[2], -1), order="C")

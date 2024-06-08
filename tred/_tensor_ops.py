"""Module with mathematical operations"""


from functools import reduce

import numpy as np

from ._m_transforms import generate_default_m_transform_pair
from ._utils import _singular_vals_tensor_to_mat, _singular_vals_mat_to_tensor

# np.einsum('mpi,pli->mli', tens1, tens2)
# the following is a quicker version of the above using numpy broadcasting
_BINARY_FACEWISE = lambda tens1, tens2: (
    tens1.transpose(2, 0, 1) @ tens2.transpose(2, 0, 1)
).transpose(1, 2, 0)


def facewise_product(*tensors):
    """Compute cumulative facewise product.

    Definition:
    $$
        C_{:,:,i} = A_{:,:,i} B_{:,:,i}
    $$

    Parameters
    ----------
    *tensors : ndarray
        Variable number of tensors, such that all adjacent input tensors have
        shape (a, b, d) and shape (b, c, d) respectively

    Returns
    -------
    C : ndarray of shape (a, c, d)
        Facewise tensor product
    """
    # apply the lambda function cumulatively over the tensor inputs
    return reduce(_BINARY_FACEWISE, tensors)


def m_product(*tensors, **transforms):
    """Kilmer et al. (2021) tensor m-product for order-3 tensors.

    See paper for definition.

    Parameters
    ----------
    *tensors : ndarray
        Variable number of tensors, such that all adjacent input tensors have
        shape (a, b, d) and shape (b, c, d) respectively

    M : Callable[[ndarray], ndarray] or None, default=None
        A function which, given some order-3 tensor, returns it under an
        orthogonal tubal transformation

    MInv : Callable[[ndarray], ndarray] or None, default=None
        A function implementing the inverse tubal transformation of M

    Returns
    -------
    m_product : ndarray of shape (a, c, d)
        Tensor-tensor m-product as found in Kilmer et al. (2021)

    References
    ----------
    Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data.
    Proceedings of the National Academy of Sciences, 118(28), p.e2015851118.
    """
    # process the m-transform keyword inputs, applying defaults if needed
    default_transforms = {"M": None, "Minv": None}
    transforms = {**default_transforms, **transforms}
    assert not (
        callable(transforms["M"]) ^ callable(transforms["Minv"])
    ), "If explicitly defined, both M and its inverse must be defined"

    if not callable(transforms["M"]):
        M, Minv = generate_default_m_transform_pair(tensors[0].shape[-1])
    else:
        M, Minv = transforms["M"], transforms["Minv"]

    # use a generator representing all the tensors under the m-transform
    # ('hat space'), allowing it to be lazily reduced over using the binary
    # facewise product anonymous function. prevents storing all of the
    # transformed tensors in memory at the same time
    return Minv(reduce(_BINARY_FACEWISE, (M(tens) for tens in tensors)))


def tsvdm(
    A,
    M=None,
    Minv=None,
    *,
    keep_hats=False,
    full_frontal_slices=True,
    svals_matrix_form=False,
):
    """Return the t-SVDM decomposition from Kilmer et al. (2021).

    This is a modified version, inspired by the implementation at
    https://github.com/UriaMorP/mprod_package

    *NOTE: For now, unlike some other implementations (Numpy, Scipy), we
    will return the tensor $V$ NOT $V^T$.*

    Parameters
    ----------
    A : ndarray of shape (n, p, t)
        Data tensor

    M : Callable[[ndarray], ndarray] or None, default=None
        A function which expects an order-3 tensor as input, and returns the
        image under a m-transform. If unspecified TPCA will use the Discrete
        Cosine Transform (ii) from `scipy.fft`.

    MInv : Callable[[ndarray], ndarray] or None, default=None
        A function implementing the inverse transform of `M`.

    keep_hats : bool, default=False
        Setting to `True` will return the tSVDM factors in the m-transform
        space, under the specified `M`

    full_frontal_slices : bool, default=True
        To reconstruct `A`, one only needs the first $k$ columns of
        $U$ and $V$.

        Setting this to False will return the truncated tensors.
        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

    svals_matrix_form : bool, default=False
        Setting to `True` will return a compressed version of $S$, where the
        singular values of each f-diagonal frontal slice becomes the column of
        a matrix, with `t` columns total.

    Returns
    -------
    U_tens : ndarray of shape (n, n, t)
        If `full_frontal_slices==False` shape is (n, k, t) instead

    S_tens : ndarray of shape (n, p, t)
        If `full_frontal_slices==False` shape is (k, k, t) instead
        If `svals_matrix_form==True`, `S_mat` of shape (k, t) returned instead

    V_tens : ndarray of shape (p, p, t)
        If `full_frontal_slices==False` shape is (p, k, t) instead

    References
    ----------
    Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data.
    Proceedings of the National Academy of Sciences, 118(28), p.e2015851118.
    """

    assert len(A.shape) == 3, "Ensure order-3 tensor input"
    assert not (
        callable(M) ^ callable(Minv)
    ), "If explicitly defined, both M and its inverse must be defined"

    if not callable(M):  # and Minv is not defined - guaranteed by assertion
        M, Minv = generate_default_m_transform_pair(A.shape[-1])

    # transform the tensor to new space via the mode-3 product
    hatA = M(A)

    # an appropriate transposition allows Numpys array broadcasting to do
    # facewise svd's S_mat contains the singular values per matrix in the
    # input stack of matrices (the transpose tensor stacks top to bottom,
    # with t slices of size n by p)
    U_stack, S_mat, Vt_stack = np.linalg.svd(
        hatA.transpose(2, 0, 1), full_matrices=full_frontal_slices
    )

    hatU = U_stack.transpose(1, 2, 0)
    S_mat = S_mat.transpose()
    # the following is a call to .transpose(1, 2, 0) followed by a facewise
    # transpose defined by .transpose(1, 0, 2)
    hatV = Vt_stack.transpose(2, 1, 0)

    # if we are transforming scipy's singular values matrix back into tensor
    # form, make sure we use the correct dimensions corresponding to whether
    # or not the tensor faces were truncated during svd
    if not svals_matrix_form:
        if full_frontal_slices:
            desired_S_tens_shape = A.shape
        else:
            n, p, t = A.shape
            k = min(n, p)
            desired_S_tens_shape = (k, k, t)

    if keep_hats:
        return (
            hatU,
            # by default return S as n,p,t f-diagonal tensor (or) convert into
            # compressed matrix of singular values of shape (k,t)
            S_mat
            if svals_matrix_form
            else _singular_vals_mat_to_tensor(S_mat, *desired_S_tens_shape),
            hatV,
        )
    else:
        return (
            Minv(hatU),
            # by default return S as n,p,t f-diagonal tensor (or) convert into
            # compressed matrix of singular values of shape (k,t)
            Minv(S_mat)
            if svals_matrix_form
            else _singular_vals_mat_to_tensor(Minv(S_mat), *desired_S_tens_shape),
            Minv(hatV),
        )


def _rank_q_truncation_zero_out(hatU, hatS, hatV, *, q=None, sigma_q=None):
    """In-place explicit rank-q truncation as introduced in Mor et al. (2022).

    Truncates (by zero-ing out) the tensors $U$, $S$, $V$ from a `tsvdm`
    decomposition to achieve an explicit rank of q.

    Parameters
    ----------
    U : ndarray, shape (n, k, t)
        `U_tens` from the tsvdm.

    S : ndarray, shape (k, k, t) or (k, t)
        `S_tens` from the tsvdm, or `S_mat` if in compact matrix form.

    V : ndarray, shape (p, k, t)
        `V_tens` from the tSVDM.

    q : int or None, default=None
        Target explicit rank for the truncation.

    sigma_q : float or None, default=None
        The `q`-th largest singular value, if known in the calling state
        already. This will not be checked.

        If `sigma_q` is set, then the `q` input parameter will be ignored.
        Saves re-computation of the `q`-th largest singular value.

    Returns
    -------
    rho : ndarray, shape (q,)
        The multi-rank which results from the choice of `q` (or `sigma_q`).

        See Mor et al. (2022)

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
    # np.searchsorted is called on each column, reversed so that it is
    # ascending, where it returns the index i
    #   s.t. sigma_q <= ascending_col[i] (see numpy searchsorted docs)
    # which mean that we want to discard the i smallest values, and keep the
    # other k-i, which is the number we store in rho
    rho = np.apply_along_axis(
        lambda col: k - np.searchsorted(col[::-1], sigma_q, side="left"),
        axis=0,
        arr=hatS,
    )

    # perform multi-rank truncation, t should be modestly sized, so the
    # for-loop should be bearable
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
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications.
    SIAM review, 51(3), pp.455-500.
    """
    # unfold the n x p x t tensor into a n x pt 2d array (matrix), where each
    # frontal slice sits 'next' each other.
    #
    # NUMPY NOTES:
    # first transpose into a n-vertical-stack of t x p matrices. reshape is
    # equivalent to 'ravel'-ling first, before reshaping the vector (using the
    # same memory format) into the intended shape. in "C" memory format, the
    # last indexes move the quickest, so ravel moves along the p index, and
    # then the t index, and then the n index. when placing this into the
    # matrix form, we get the intended result, where for the same t and n, the
    # p values sit next to each other; for the same n, all of its t values sit
    # together on the same row. then the matrix shape ensures that each row
    # has a distinct n
    #
    # the unfolding has the format as singular_values_ from the fit method
    if view:
        return tens.view().transpose(0, 2, 1).reshape((tens.shape[0], -1), order="C")
    else:
        return tens.copy().transpose(0, 2, 1).reshape((tens.shape[0], -1), order="C")


def _mode_2_unfold(tens, view=False):
    """Return mode-2 unfolding copy, as defined in Kolda et al.

    References
    ----------
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications.
    SIAM review, 51(3), pp.455-500.
    """
    if view:
        return tens.view().transpose(1, 2, 0).reshape((tens.shape[1], -1), order="C")
    else:
        return tens.copy().transpose(1, 2, 0).reshape((tens.shape[1], -1), order="C")


def _mode_3_unfold(tens, view=False):
    """Return mode-3 unfolding copy, as defined in Kolda et al.

    References
    ----------
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications.
    SIAM review, 51(3), pp.455-500.
    """
    if view:
        return tens.view().transpose(2, 1, 0).reshape((tens.shape[2], -1), order="C")
    else:
        return tens.copy().transpose(2, 1, 0).reshape((tens.shape[2], -1), order="C")

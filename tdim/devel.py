"""
In development...
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import dct, idct


def _default_transform(t):
    """Wrapper around scipy fft to generate functions to perform $\times_3 M$ operations
    on order-3 tensors, where M is defined by the (scaled) Discrete Cosine Transform.

    See [1] and [2].

    Parameters
    ----------
        t: int
            The length of the transform

    Returns
    -------
        fun_m: Callable[[ArrayLike], ArrayLike]
            A function which expects an order-3 tensor as input, and applies DCT-II to
            each of the tubal fibres. This preserves the dimensions of the tensor.

        inv_m: Callable[[ArrayLike], ArrayLike]
            A tensor transform (the inverse of `fun_m`)
    """

    def M(x):
        # we expect the third dimensions of the tensor to match the transform length
        assert len(x.shape) == 3, "Expecting order 3 tensor input"
        assert x.shape[-1] == t, f"Expecting input shape to be ?x?x{t}"

        return dct(x, type=2, n=t, axis=-1, norm="ortho")

    def Minv(x):
        # we expect the third dimensions of the tensor to match the transform length
        assert len(x.shape) == 3, "Expecting order 3 tensor input"
        assert x.shape[-1] == t, f"Expecting input shape to be ?x?x{t}"

        return idct(x, type=2, n=t, axis=-1, norm="ortho")

    return M, Minv


def _singular_vals_mat_to_tensor(mat, n, p, t):
    """Decommpress $k \times t$ matrix of singular values into $\hat{S}$ from literature.

    Parameters
    ----------
        mat: ArrayLike
            $k \times t$ matrix representation

        n: int
            first dimension of output tensor

        p: int
            second dimension of output tensor

        t: int
            third dimensions of output tensor

    Returns
    -------
        hatS: ArrayLike
            $n \times p \times t$ tensor representation
    """
    k = min(n, p)
    assert k == mat.shape[0] and t == mat.shape[1]

    hatS = np.zeros((n, p, t))
    for i in range(t):
        hatS[:k, :k, i] = mat[:, i]

    return hatS


def _singular_vals_tensor_to_mat(tensor):
    """Compresses $\hat{S}$, a $n \times p \times t$ f-diagonal tensor of singular values
    into a $k \times t$ matrix. Each row contains the $k$ singular values from that
    longitudinal slice.

    Parameters
    ----------
        tensor: ArrayLike
            $n \times p \times t$ tensor representation

    Returns
    -------
        mat: ArrayLike
            $k \times t$ matrix representation of singular values, where
            $k = \min{(n,p)}$
    """
    n, p, t = tensor.shape
    k = min(n, p)
    mat = np.zeros((k, t))
    for i in range(t):
        mat[:, i] = np.diagonal(tensor[:, :, i])

    return mat


def tSVDM(A, M, Minv, *, hats=False, full_matrices=False, compact_svals=False):
    """Return the t-SVDM decomposition from Kilmer et al. (2021). Currently, this is
    basically identical to the implementation in https://github.com/UriaMorP/mprod_package
    but we plan to update this in future, potentially adopting the TensorFlow framework.

    See [1]

    Parameters
    ----------

    Returns
    -------
    """

    # transform the tensor to new space via the mode-3 product
    hatA = M(A)

    # an appropriate transposition allows Numpys array broadcasting to work appropriately
    # S_mat contains the singular values per matrix in the input matrix 'stack'
    # we reshape into a sparse tensor
    #
    # the transpose tensor stacks top to bottom, with t horizontal slices of size n by p
    U_stack, S_mat, V_stack = np.linalg.svd(
        hatA.transpose(2, 0, 1), full_matrices=full_matrices
    )

    hatU = U_stack.transpose(1, 2, 0)
    hatV = V_stack.transpose(2, 1, 0)

    if hats:
        # by default, convert into n,p,t tensor 
        # (optionally) compressed matrix of singular values of size k,t
        hatS = (
            _singular_vals_mat_to_tensor(S_mat.transpose(), *A.shape)
            if not compact_svals
            else S_mat.transpose()
        )
        return hatU, hatS, hatV
    else:
        S = Minv(_singular_vals_mat_to_tensor(S_mat.transpose(), *A.shape))
        S = _singular_vals_tensor_to_mat(S) if compact_svals else S
        U, V = Minv(hatU), Minv(hatV)
        return U, S, V

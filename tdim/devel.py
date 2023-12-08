"""
In development...
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import dct, idct
from sklearn.base import BaseEstimator, TransformerMixin


def m_product(A, B, M, Minv):
    """Kilmer et al. (2021) tensor m-product for order-3 tensors. See [1]

    Parameters
    ----------

    Returns
    -------
    """
    assert (
        A.shape[1] == B.shape[0] and A.shape[2] == B.shape[2]
    ), "Non conforming dimensions"

    hatA, hatB = M(A), M(B)

    # hatC = np.einsum('mpi,pli->mli', hatA, hatB)
    # the following is a quicker version of the above using numpy broadcasting
    hatC = np.matmul(hatA.transpose(2, 0, 1), hatB.transpose(2, 0, 1)).transpose(1, 2, 0)
    
    return Minv(hatC)


def _default_transform(t):
    """Wrapper around scipy fft to generate functions to perform $\times_3 M$ operations
    on order-3 tensors, where M is defined by the (scaled) Discrete Cosine Transform.

    As introduced by Kilmer et al. (2021) and applied by Mor et al. (2022)

    Parameters
    ----------
        t : int
            The length of the transform

    Returns
    -------
        fun_m : Callable[[ArrayLike], ArrayLike]
            A function which expects an order-3 tensor as input, and applies DCT-II to
            each of the tubal fibres. This preserves the dimensions of the tensor.

        inv_m : Callable[[ArrayLike], ArrayLike]
            A tensor transform (the inverse of `fun_m`)
    """

    def M(x):
        # we expect the third dimensions of the tensor to match the transform length
        assert len(x.shape) == 3, "Expecting order-3 tensor input"
        assert x.shape[-1] == t, f"Expecting input dimensions to be ?x?x{t}"

        # allows parallel workers up to os.cpu_count. See scipy fft for more details
        return dct(x, type=2, n=t, axis=-1, norm="ortho", workers=-1)

    def Minv(x):
        # we expect the third dimensions of the tensor to match the transform length
        assert len(x.shape) == 3, "Expecting order-3 tensor input"
        assert x.shape[-1] == t, f"Expecting input dimensions to be ?x?x{t}"

        # allows parallel workers up to os.cpu_count. See scipy fft for more details
        return idct(x, type=2, n=t, axis=-1, norm="ortho", workers=-1)

    return M, Minv


def _singular_vals_mat_to_tensor(mat, n, p, t):
    """Decompress $k \times t$ matrix of singular values into $\hat{S}$ from literature.

    Parameters
    ----------
        mat : ArrayLike, shape: (k, tt)
            $k \times t$ matrix representation, the function checks if `tt == t`

        n : int
            first dimension of output tensor

        p : int
            second dimension of output tensor

        t : int
            third dimensions of output tensor

    Returns
    -------
        hatS : ArrayLike, shape: (n, p, t)
            $n \times p \times t$ tensor representation
    """
    k = min(n, p)
    assert k == mat.shape[0] and t == mat.shape[1], "Ensure conforming dimensions"

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
        tensor : ArrayLike, shape: (n, p, t)
            $n \times p \times t$ tensor representation

    Returns
    -------
        mat : ArrayLike, shape: (min(n, p), t)
            $k \times t$ matrix representation of singular values, where
            $k = \min{(n,p)}$
    """
    n, p, t = tensor.shape
    k = min(n, p)
    mat = np.zeros((k, t))
    for i in range(t):
        mat[:, i] = np.diagonal(tensor[:, :, i])

    return mat


def tSVDM(A, M, Minv, *, keep_hats=False, full_matrices=True, compact_svals=False):
    """Return the t-SVDM decomposition from Kilmer et al. (2021). Currently, this is
    a modified version of the implementation at https://github.com/UriaMorP/mprod_package
    but we plan to update this in future, potentially adopting the TensorFlow framework.

    NOTE: For now, unlike some other implementations (Numpy), we will return the tensor
    $V$ NOT $V^T$. However, transposes are well-defined under this tensor-product
    framework. See [1]

    Parameters
    ----------
        A : ArrayLike, shape: (n, p, t)
            $n \times p \times t$ data tensor

        M : Callable[[ArrayLike], ArrayLike]
            A function which, given some order-3 tensor, returns it under some $\times_3$
            invertible transformation.

        MInv : Callable[[ArrayLike], ArrayLike]
            The inverse transformation of M

        keep_hats : bool, default=False
            Setting to `True` will return the tSVDM factors in the tensor domain transform
            space, under the specified $M$

        full_matrices : bool, default=True
            In practice, one only needs the first $k$ columns of $U_{:,:,i}$, $V_{:,:,i}$.
            Setting this to False will return tensors truncated, by removing columns after
            the k-th one in U or V.
            See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

        compact_svals : bool, default=False
            Setting to `True` will return a compressed version of $S$, whereby the
            singular values of each f-diagonal frontal slice becomes the column of a
            matrix, with t columns total

    Returns
    -------
        U_tens : ArrayLike, shape: (n, n, t) if full_matrices else (n, k, t)

        S_tens : ArrayLike, shape: (n, p, t) if not compact_svals else (k, t)

        V_tens : ArrayLike, shape: (p, p, t) if full_matrices else (p, k, t)
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

    if keep_hats:
        # by default return S as n,p,t f-diagonal tensor, matching literature
        # (optionally) convert into compressed matrix of singular values of size k,t
        hatS = (
            _singular_vals_mat_to_tensor(S_mat.transpose(), *A.shape)
            if not compact_svals
            else S_mat.transpose()
        )
        return hatU, hatS, hatV
    else:
        # regardless of compact_svals setting, S needs to be converted into tensor form
        # for Minv to be applied
        # NOTE: there may be a more efficient way to do this
        S = Minv(_singular_vals_mat_to_tensor(S_mat.transpose(), *A.shape))
        S = _singular_vals_tensor_to_mat(S) if compact_svals else S
        U, V = Minv(hatU), Minv(hatV)
        return U, S, V


class TCAM(BaseEstimator, TransformerMixin):
    """Mor et al. (2020) t-SVDM tensor analogue of PCA using explicit rank truncation

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, M, Minv):
        self.M = M
        self.Minv = Minv

    def _mprod(self, A, B):
        pass

"""
In development...
We want to give appropriate credit to https://github.com/UriaMorP/mprod_package; we are 
rewriting the key implementations ourselves to better suit our purposes, and future 
development we are interested in.
"""


from numbers import Integral, Real

import numpy as np
import os
from scipy.fft import dct, idct
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_scalar


class RealNotInt(Real):
    """A type that represents reals that are not instances of int.

    Behaves like float, but also works with values extracted from numpy arrays.
    isintance(1, RealNotInt) -> False
    isinstance(1.0, RealNotInt) -> True

    From sklearn/utils/_param_validation.py
    """


RealNotInt.register(float)


def _facewise_product(A, B):
    """Facewise product s.t. $C_{:,:,i} = A_{:,:,i} B{:,:,i}$

    This could be optimized later if needed.

    Parameters
    ----------
    A : ArrayLike
        $a \times b \times d$ tensor representation

    B : ArrayLike
        $b \times c \times d$ tensor representation

    Returns
    -------
    C : ArrayLike
        $a \times c \times d$ tensor representation

    """

    # return np.einsum('mpi,pli->mli', A, B)
    # the following is a quicker version of the above using numpy broadcasting
    return np.matmul(A.transpose(2, 0, 1), B.transpose(2, 0, 1)).transpose(1, 2, 0)


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
    return Minv(_facewise_product(hatA, hatB))


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
        fun_m : Callable[[ArrayLike], ArrayLike]
            A function which expects an order-3 tensor as input, and applies DCT-II to
            each of the tubal fibres. This preserves the dimensions of the tensor.

        inv_m : Callable[[ArrayLike], ArrayLike]
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
        hatS : ArrayLike, shape: (n, p, t)
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
        mat : ArrayLike, shape: (min(n, p), t)
            $k \times t$ matrix representation of singular values, where
            $k = \min{(n,p)}$
    """
    n, p, t = tensor.shape
    k = min(n, p)
    S_mat = np.zeros((k, t))
    for i in range(t):
        S_mat[:, i] = np.diagonal(tensor[:, :, i])

    return S_mat


def tsvdm(
    A,
    M=None,
    Minv=None,
    *,
    keep_hats=False,
    full_frontal_slices=True,
    svals_matrix_form=False,
):
    """Return the t-SVDM decomposition from Kilmer et al. (2021). Currently, this is
    a modified version of the implementation at https://github.com/UriaMorP/mprod_package
    but we plan to update this in future, potentially adopting the TensorFlow framework,
    or adopt other matrix svd implementations.

    NOTE: For now, unlike some other implementations (Numpy, Scipy), we will return the
    tensor $V$ NOT $V^T$.

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

        full_frontal_slices : bool, default=True
            In practice, one only needs the first $k$ columns of $U_{:,:,i}$, $V_{:,:,i}$.
            Setting this to False will return tensors truncated, by removing columns after
            the k-th one in U or V.
            See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

        svals_matrix_form : bool, default=False
            Setting to `True` will return a compressed version of $S$, whereby the
            singular values of each f-diagonal frontal slice becomes the column of a
            matrix, with t columns total

    Returns
    -------
        U_tens : ArrayLike, shape: (n, n, t) if full_matrices else (n, k, t)

        S_tens : ArrayLike, shape: (n, p, t) if not compact_svals else (k, t)

        V_tens : ArrayLike, shape: (p, p, t) if full_matrices else (p, k, t)
    """

    assert not (
        callable(M) ^ callable(Minv)
    ), "If explicitly defined, both M and its inverse must be defined"

    if M is None:  # and Minv is None - guaranteed by assertion
        M, Minv = generate_DCTii_M_transform_pair(A.shape[2])

    # transform the tensor to new space via the mode-3 product
    hatA = M(A)

    # an appropriate transposition allows Numpys array broadcasting to work appropriately
    # S_mat contains the singular values per matrix in the input matrix 'stack'
    # we reshape into a sparse tensor
    #
    # the transpose tensor stacks top to bottom, with t horizontal slices of size n by p
    U_stack, S_mat, V_stack = np.linalg.svd(
        hatA.transpose(2, 0, 1), full_matrices=full_frontal_slices
    )

    hatU = U_stack.transpose(1, 2, 0)
    S_mat = S_mat.transpose()
    hatV = V_stack.transpose(2, 1, 0)

    if keep_hats:
        return (
            hatU,
            # by default return S as n,p,t f-diagonal tensor, matching literature
            # (optionally) convert into compressed matrix of singular values of size k,t
            S_mat
            if svals_matrix_form
            else _singular_vals_mat_to_tensor(S_mat, *A.shape),
            hatV,
        )
    else:
        return (
            Minv(hatU),
            # by default return S as n,p,t f-diagonal tensor, matching literature
            # (optionally) convert into compressed matrix of singular values of size k,t
            Minv(S_mat)
            if svals_matrix_form
            else _singular_vals_mat_to_tensor(Minv(S_mat), *A.shape),
            Minv(hatV),
        )


class TPCA(BaseEstimator, TransformerMixin):
    """t-SVDM tensor analogue of PCA using explicit rank truncation with explicit rank
    truncation from Mor et al. (2022), and underlying m-product framework from Kilmer et
    al. (2021).

    In the future, different svd solvers and truncation methods will likely be implemented
    here.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, n_components=None, copy=True, M=None, Minv=None):
        assert not (
            callable(M) ^ callable(Minv)
        ), "If explicitly defined, both M and its inverse must be defined"

        if M is None:  # and Minv is None - guaranteed by assertion
            self._default_transform_pair_generator = staticmethod(
                generate_DCTii_M_transform_pair
            )
        else:
            self._default_transform_pair_generator = None

        self.n_components = n_components
        self.copy = copy

        # once the user sets the transform, we do not want them to change it as a public
        # attribute
        self._M = M
        self._Minv = Minv

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X. This mirrors
        sklearn's PCA fit_transform method, whereby $Z = X \times_M V = U \times_M S$

        """
        hatU, hatS_mat, hatV = self._fit(X)
        hatU = hatU

        """
        ... in progress ...
        """

        return hatU

    def _fit(self, X):
        """Fit the model by computing the full SVD on X
        Implementation loosely modelled around _fit_full() method from sklearn's PCA.

        In the future, we could potentially explore different svd solvers which lets one
        directly pass truncation specifications into the low-level solver...?
        """
        assert len(X.shape) == 3, "Ensure order-3 tensor input"
        n, p, t = X.shape
        k = min(n, p)

        # center the data into mean deviation form
        # similar to sklearns PCA, we choose to implement this within the class and just
        # store the mean slice for subsequent calls to transform
        self.mean_ = np.mean(X, axis=0)
        if self.copy:
            X = X.copy()
        X -= self.mean_

        # if there is no explicitly defined transform in __init__, assign functions to
        # perform a default transformation
        if not self._default_transform_pair_generator is None:
            self._M, self._Minv = self._default_transform_pair_generator.__func__(
                X.shape[2]
            )

        # perform tensor decomposition via Kilmer's tSVDM
        hatU, hatS_mat, hatV = tsvdm(
            X,
            self._M,
            self._Minv,
            keep_hats=True,
            full_frontal_slices=False,
            svals_matrix_form=True,
        )

        # get variance explained by singular values
        squared_singular_values = (hatS_mat**2).ravel()
        total_var = np.sum(squared_singular_values)
        explained_variance_ratio_ = squared_singular_values / total_var

        # process n_components input
        if self.n_components is None:
            n_components = k * t
        elif isinstance(self.n_components, RealNotInt):
            if 0 < self.n_components < 1.0:
                ratio_cumsum = np.cumsum(explained_variance_ratio_)
                n_components = (
                    np.searchsorted(ratio_cumsum, n_components, side="right") + 1
                )
            else:
                raise ValueError(
                    "For non-integer inputs, ensure that 0 < n_components < 1"
                )
        elif isinstance(self.n_components, Integral):
            if 0 <= self.n_components <= k:
                n_components = self.n_components
            else:
                raise ValueError(
                    f"For integer inputs, ensure that 0 <= n_components <= min(n, p)={k}"
                )
        else:
            raise TypeError(
                "n_components must be an integer, float, or None"
                f"Got {type(self.n_components)} instead"
            )

        # as per sklearn conventions, attributes with a trailing underscore indicate that
        # they have been stored following a call to fit()
        self.n_, self.p_, self.t_ = n, p, t
        self.n_components_ = n_components

        return hatU, hatS_mat, hatV

    def _transform(self, X):
        assert len(X.shape) == 3, "Ensure order-3 tensor input"
        assert (
            X.shape[1] == self.p_ and X.shape[2] == self.t_
        ), "Ensure the number of features, and time points, matches the model fit data"

        # center the data, ensure we do not do this in-place using -=
        X = X - self.mean_

        # in the interest of efficiency, we returned V in the m-transformed space from
        # tsvdm saving two redundant calls to M and Minv
        # but, the code below is a bit ugly(ier) than a straightforward call to
        # m_product()
        X_transformed = self._Minv(_facewise_product(self._M(X), self.hat_components_))

        return X_transformed

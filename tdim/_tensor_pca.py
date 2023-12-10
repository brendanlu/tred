"""In development...
We want to give appropriate credit to https://github.com/UriaMorP/mprod_package; we are 
rewriting the key implementations ourselves to better suit our purposes, and future 
development we are interested in.
"""


from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._base import RealNotInt, _facewise_product, m_product
from ._utils import (
    generate_DCTii_M_transform_pair,
    _singular_vals_mat_to_tensor,
    _singular_vals_tensor_to_mat,
)


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

        M : Callable[[ArrayLike], ndarray]
            A function which, given some order-3 tensor, returns it under some $\times_3$
            invertible transformation.

        MInv : Callable[[ArrayLike], ndarray]
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
        U_tens : ndarray, shape: (n, n, t) if full_matrices else (n, k, t)

        S_tens : ndarray, shape: (n, p, t) if not compact_svals else (k, t)

        V_tens : ndarray, shape: (p, p, t) if full_matrices else (p, k, t)
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
    U_stack, S_mat, Vt_stack = np.linalg.svd(
        hatA.transpose(2, 0, 1), full_matrices=full_frontal_slices
    )

    hatU = U_stack.transpose(1, 2, 0)
    S_mat = S_mat.transpose()
    # the following is a call to .transpose(1, 2, 0) followed by a facewise transpose
    # defined by .transpose(1, 0, 2)
    hatV = Vt_stack.transpose(2, 1, 0)

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

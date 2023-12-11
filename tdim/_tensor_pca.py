"""In development...
We want to give appropriate credit to https://github.com/UriaMorP/mprod_package; we are 
rewriting the key implementations ourselves to better suit our purposes, and future 
development we are interested in.
"""


from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._base import RealNotInt, _facewise_product, _m_product
from ._utils import _singular_vals_mat_to_tensor, _singular_vals_tensor_to_mat
from ._m_transforms import generate_DCTii_M_transform_pair


def _generate_default_m_transform_pair(t):
    """Wrapper for a function that generates functions M and Minv given the length of
    a tubal fibre.

    Both the tsvdm and TPCA do not require the user to explicitly specify a tensor
    m-transform. In those cases, they rely on this function here.

    We may choose to choose a different default family of mappings later

    Parameters
    ----------
        t : int
            The length of the transform (length of the tubal fibers)

    Returns
    -------
        fun_m : Callable[[ArrayLike], ndarray]
            A function which expects an order-3 tensor as input, and applies a tensor
            transform to each of the tubal fibres. This preserves the dimensions of
            the tensor.

        inv_m : Callable[[ArrayLike], ndarray]
            A tensor transform (the inverse of `fun_m`)
    """

    # as introduced by Kilmer et al. (2021) and applied by Mor et al. (2022), we currently
    # use the Discrete Cosine transform
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
    return generate_DCTii_M_transform_pair(t)


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
        M, Minv = _generate_default_m_transform_pair(A.shape[2])

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

    In the future, different svd solvers and truncation methods will likely be
    implemented here.

    Mirrors API and implementation of sklearn/decomposition/_pca.py for matrix data, as
    much as possible.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, n_components=None, copy=True, *, M=None, Minv=None):
        self.n_components = n_components
        self.copy = copy
        self.M = M
        self.Minv = Minv

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

    def _fit_transform_naive(self, X, y=None):
        self._fit(X)
        return self._transform(X)

    def _fit(self, X):
        """Fit the model by computing the full SVD on X
        Implementation loosely modelled around _fit_full() method from sklearn's PCA.

        In the future, we could potentially explore different svd solvers which lets one
        directly pass truncation specifications into the low-level solver...?
        """

        assert not (
            callable(self.M) ^ callable(self.Minv)
        ), "If explicitly defined, both M and its inverse must be defined"

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
        if self.M is None:  # and Minv is None - guaranteed by assertion
            self.M, self.Minv = _generate_default_m_transform_pair(X.shape[2])

        # perform tensor decomposition via Kilmer's tSVDM
        hatU, hatS_mat, hatV = tsvdm(
            X,
            self.M,
            self.Minv,
            keep_hats=True,
            full_frontal_slices=False,
            svals_matrix_form=True,
        )

        # get variance explained by singular values
        # note that we are not yet aware of any notion of 'variance' for random tensors
        # so we do not have sklearn PCA's self.explained_variance_
        # however we may find literature for this in the future to include it
        singular_values_ = hatS_mat.copy().ravel()
        singular_values_[::-1].sort()  # sort decreasing order in place, avoiding copy
        squared_singular_values = singular_values_**2
        total_var = np.sum(squared_singular_values)
        explained_variance_ratio_ = squared_singular_values / total_var

        # process n_components input
        if self.n_components is None:
            n_components = k * t
        elif isinstance(self.n_components, RealNotInt):
            if 0 < self.n_components < 1.0:
                # retrieve the integer number of components required to explain the a
                # greater proportion of the total squared sum of singular values
                ratio_cumsum = np.cumsum(explained_variance_ratio_)
                # np.searchsorted call returns the mininum i
                #   s.t. n_components < ratio_cumsum[i] (see docs)
                # which means that the (i+1)th element in ratio_cumsum[i] strictly exceeds
                # the user's specified variance ratio
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

        # q := n_components
        # this is the smallest singular value in the truncation
        # EDGE CASE 1: singular_values_[n_components-1] == singular_values_[n_components]
        sigma_q = singular_values_[n_components - 1]

        # now we compute a representation of rho, of size t; see Mor et al. (2022)
        # each diagonal of the frontal slice is already sorted in decreasing order
        # np.searchsorted is called on each column, reversed so that it is ascending,
        # where it returns the index i
        #   s.t. sigma_q <= ascending_col[i]
        # which mean that we want to discard the i smallest values, and keep the other
        # k-i, which is the number we store in rho
        rho = np.apply_along_axis(
            lambda col: k - np.searchsorted(col[::-1], sigma_q, side="left"),
            axis=0,
            arr=hatS_mat,
        )
        # add a funny error message for EDGE CASE 1, for now...
        assert np.sum(rho) == n_components, (
            "Ah, you are the first unlucky person to run into a rare edge case"
            "This is because your singular values tensor contains tied singular values"
            "Our implementation is yet to appropriately handle this"
            "Let us know and we will implement the logic, if we have not already"
        )

        # perform multi-rank truncation, t should be modestly sized so the for-loop
        # should be bearable
        for i in range(t): 
            hatU[:, :rho[i], i] = 0
            hatS_mat[:rho[i], i] = 0 
            hatV[:, :rho[i], i] = 0

        # store useful attributes; as per sklearn conventions, we use trailing underscores 
        # to indicate that they have been populated following a call to fit()
        self.n_, self.p_, self.t_ = n, p, t
        self.n_components_ = n_components
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

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
        X_transformed = self.Minv(_facewise_product(self.M(X), self.hat_components_))

        return X_transformed

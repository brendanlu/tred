"""Tensor Component Analysis based on the TCAM algorithm and tensor m-product.

We want to give appropriate credit to https://github.com/UriaMorP/mprod_package for 
existing implementations of algorithms in this module; we are rewriting the key 
implementations ourselves to better suit our purposes, and future development we are 
interested in.
"""


from numbers import Integral
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted

from ._tensor_ops import _facewise_product
from ._utils import RealNotInt, _singular_vals_mat_to_tensor
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
        U_tens : ndarray, shape: (n, n, t) if full_frontal_slices==True else (n, k, t)

        S_tens : ndarray, shape: (n, p, t) if svals_matrix_form==False else (k, t)

        V_tens : ndarray, shape: (p, p, t) if full_frontal_slices==True else (p, k, t)
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
    # (the transpose tensor stacks top to bottom, with t horizontal slices of size n by p)
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


class TPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """t-SVDM tensor analogue of PCA using explicit rank truncation with explicit rank
    truncation from Mor et al. (2022), and underlying m-product framework from Kilmer et
    al. (2021). Takes in an $n \times p \times t$ input tensor, and transforms into
    a $n \times$ `n_components` matrix of 2D transformed loadings.

    The input tensor is centred into Mean Deviation Form (by location), but not normalized
    (by scale).

    Parameters
    ----------
    n_components : int, float, or None, default=None
        Control number of components to keep. If n_components is not set at all, or set
        to `None`, ``n_components == min(n, p) * t``

        If `n_components` is an integer, the TPCA will return the number of loadings,
        provided that for the tensor data passed into `fit`, satisfies
        ``1 <= n_components <= min(n, p) * t``

        If ``0 < n_components < 1``, TPCA will select the number of compononents such that
        the amount of variance that needs to be explained is greater than the percentage
        specified.

    copy : bool, default=True
        If False, data passed to fit are overwritten and running fit(X).transform(X) will
        not yield the expected results. Use fit_transform(X) instead.

    M : Callable[[ArrayLike], ndarray] or None, default=None
        A function which, given some order-3 tensor, returns it under some $\times_3$
        invertible transformation. If unspecified TPCA will use the Discrete Consine
        Transform (ii) from scipy fft.

    MInv : Callable[[ArrayLike], ndarray] or None, default=None
        The inverse transformation of M.

    centre : bool, default=True
        If False, the data tensor will not be centralized into Mean Deviation Form.
        By default, the mean horizontal slice of the tensor is subtracted, so that all of
        the horizontal slices sum to 0, analagous to centering the data in PCA.

    Attributes
    ----------
    n_, p_, t_, k_ : int
        The dimensions of the training data. ``k_ == min(n_, p_)``

    n_components_ : int
        The estimated number of components. If `n_components` was explicitly set by an
        integer value, this will be the same as that. If `n_components` was a number
        between 0 and 1, this number is estimated from input data. Otherwise, if not set
        (defaults to None), it will default to $k \times t$ in the training data.

    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Percentage of total variance explained by each of the selected components. The
        selected components are selected so that this is returned in descending order.

        If ``n_components`` is not set then all components are stored and the sum of this
        ratios array is equal to 1.0.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.

    mean_ : ndarray of shape(p_, t_)
        Per-feature, per-timepoint empirical mean, estimated from the training set.
        This is used to normalize any new data passed to transform(X), unless centre
        is explicitly turned off via ``centre==False`` during object instantiation.

    References
    ----------
    For the underlying m-product algebra in which the tensor analogue of SVD, and
    therefore PCA is formulated, see:
    `Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor
    algebra for optimal representation and compression of multiway data. Proceedings
    of the National Academy of Sciences, 118(28), p.e2015851118.`

    Implements the TCAM model from
    `Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron,
    H., 2022. Dimensionality reduction of longitudinal’omics data using modern
    tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.`

    See also: https://github.com/UriaMorP/mprod_package
    """

    def __init__(self, n_components=None, *, copy=True, M=None, Minv=None, centre=True):
        # as per sklearn conventions, we perform any and all parameter validation inside
        # fit, and none in __init__
        self.n_components = n_components
        self.copy = copy
        self.M = M
        self.Minv = Minv
        self.centre = centre

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : ArrayLike of shape (n, p, t)
            Training data, where `n` is the number of samples, `p` is the number of
            features, as `t` is the number of time points.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself, after being fitted.
        """
        self._fit(X)
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        See the TCAM algorithm from Mor et al. (2022)

        Parameters
        ----------
        X : ArrayLike of shape (n, p, t)
            Training data, where `n` is the number of samples, `p` is the number of
            features, as `t` is the number of time points.

        y : Ignored
            Ignored.

        Returns
        -------
        X_transformed : ndarray of shape (n, n_components)
            TCAM projections in 2D transformed space.
        """

        check_is_fitted(self)
        assert len(X.shape) == 3, "Ensure order-3 tensor input"
        assert (
            X.shape[1] == self.p_ and X.shape[2] == self.t_
        ), "Ensure the number of features, and time points, matches the model fit data"

        if self.centre:
            X = X - self.mean_

        # in the interest of efficiency, V was returned in the m-transformed space from
        # tsvdm saving a pair of roundabout calls to M and Minv, and pick out the top
        # i_q and j_q indexes, as notated in Mor et al. (2022)
        return _facewise_product(self.M(X), self._hatV)[
            :, self._k_t_flatten_sort[0], self._k_t_flatten_sort[1]
        ]

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        The output here will not be identical to calling fix(X).transform(X). But, the
        two give the same results up to machine precision. We provide a brief discussion
        of this below.

        Parameters
        ----------
        X : ArrayLike of shape (n, p, t)
            Training data, where `n` is the number of samples, `p` is the number of
            features, as `t` is the number of time points.

        y : Ignored
            Ignored.

        Returns
        -------
        X_transformed : ndarray of shape (n, n_components)
            TCAM projections in 2D transformed space.

        Notes
        -----
        The tensor m-product from Kilmer et al. (2021) has a notion of tensor inverse,
        and tensor orthogonality.

        We benchmarked an alternative approach as taken by sklearn in their PCA class. If
        we right multiply A's tSVDM by $V$ we note that it cancels the $V^T$ giving us
            $Z = A *_M V = U *_M S$
        It appears that computing the final term is more computationally efficient, even
        if we have to convert $S$ into its full (sparse) tensor representation.

        By contrast, transform(X) will simply compute
            $Z = A *_M V$

        $Z$ needs to be converted by $\times_3 M$ and 'compressed' before being returned.
        For these details see Mor et al. (2022)
        """
        # note that these tensors do NOT have full face-wise matrices
        hatU, hatS_mat, _ = self._fit(X)
        hatS = _singular_vals_mat_to_tensor(hatS_mat, self.n_, self.k_, self.t_)
        return _facewise_product(hatU, hatS)[
            :, self._k_t_flatten_sort[0], self._k_t_flatten_sort[1]
        ]

    def _fit(self, X):
        """Fit the model by computing the full SVD on X. Implementation loosely modelled
        around _fit_full() method from sklearn's PCA. In the future, we could potentially
        explore different svd solvers which lets one directly pass truncation
        specifications into the low-level solver...?
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
        if self.center:
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

        # we flatten out the compressed singular value matrix in Fortran memory style
        # (column-wise). tensor-wise, we can interpret this as stacking the diagonals
        # of each tensor face in S next to each other in the flattened array, where the
        # singular values are grouped by face
        singular_values_ = hatS_mat.flatten(order="F")
        self._k_t_flatten_sort = singular_values_.argsort()[::-1]
        singular_values_ = singular_values_[self._k_t_flatten_sort]

        # get variance explained by singular values
        # note that we are not yet aware of any notion of 'variance' for random tensors
        # so we do not have sklearn PCA's self.explained_variance_
        # however we may find literature for this in the future to include it
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
                #   s.t. n_components < ratio_cumsum[i] (see numpy searchsorted docs)
                # which means that the (i+1)th element in ratio_cumsum[i] strictly exceeds
                # the user's specified variance ratio
                n_components = (
                    np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
                )
            else:
                raise ValueError(
                    "For non-integer inputs, ensure that 0 < n_components < 1"
                )
        elif isinstance(self.n_components, Integral):
            if 1 <= self.n_components <= k * t:
                n_components = self.n_components
            else:
                raise ValueError(
                    f"Integer inputs must satisfy 1 <= n_components <= min(n, p)*t={k*t}"
                )
        else:
            raise TypeError(
                "n_components must be an integer, float, or None"
                f"Got {type(self.n_components)} instead"
            )

        # we store the features tensor, in the tSVDM decomposition, in the transforemd
        # space, saving roundabout calls to M and Minv when performing m-product with new
        # data
        self._hatV = hatV
        # now convert the argsort indexes back into the two dimensional indexes. in the
        # same tensor semantics as hatS_mat, the rows (tuple[0]) in this multindex
        # correspond to the p-dimension location, and the columns (tuple[1]) in the
        # multindex correspond to the t-dimension location. these are now the collection
        # of i_h's and j_h's in Mor et al. (2022)
        self._k_t_flatten_sort = np.unravel_index(
            self._k_t_flatten_sort[:n_components], shape=hatS_mat.shape, order="F"
        )

        # store public attributes; as per sklearn conventions, we use trailing underscores
        # to indicate that they have been populated following a call to fit()
        self.n_, self.p_, self.t_, self.k_ = n, p, t, k
        self.n_components_ = n_components
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return hatU, hatS_mat, hatV

    @property
    def _n_features_out(self):
        """Number of transformed output features.

        [CAN IGNORE]: This is an sklearn implementation detail, and does not affect any 
        of the statistical functionality in this class.
        See sklearn/decompositions/_base.py
        """
        return self.n_components_

from packaging.version import parse as parse_version

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

from tred import TPCA


# The following has been copied over from sklearn's testing practices for sparse 
# containers, with a minor modification
# ----------------------------------------------------------------------------------------
# TODO: We can consider removing the containers and importing
# directly from SciPy when sparse matrices will be deprecated.
CSR_CONTAINERS = [scipy.sparse.csr_matrix]
CSC_CONTAINERS = [scipy.sparse.csc_matrix]
COO_CONTAINERS = [scipy.sparse.coo_matrix]
LIL_CONTAINERS = [scipy.sparse.lil_matrix]
DOK_CONTAINERS = [scipy.sparse.dok_matrix]
BSR_CONTAINERS = [scipy.sparse.bsr_matrix]
DIA_CONTAINERS = [scipy.sparse.dia_matrix]

if parse_version(scipy.__version__) >= parse_version("1.8"):
    # Sparse Arrays have been added in SciPy 1.8
    # TODO: When SciPy 1.8 is the minimum supported version,
    # those list can be created directly without this condition.
    # See: https://github.com/scikit-learn/scikit-learn/issues/27090
    CSR_CONTAINERS.append(scipy.sparse.csr_array)
    CSC_CONTAINERS.append(scipy.sparse.csc_array)
    COO_CONTAINERS.append(scipy.sparse.coo_array)
    LIL_CONTAINERS.append(scipy.sparse.lil_array)
    DOK_CONTAINERS.append(scipy.sparse.dok_array)
    BSR_CONTAINERS.append(scipy.sparse.bsr_array)
    DIA_CONTAINERS.append(scipy.sparse.dia_array)
# ----------------------------------------------------------------------------------------

GLOBAL_SEED = 1

# various n, p, t sizes
# ensure n > p, p > n inputs are tested
TENSOR_SIZES = [(100, 30, 20), (20, 1000, 5)]

# test tiny, small, medium, and large numbers
ELEMENT_SCALES = [10**i for i in range(-2, 4)]

# sparse size input
# a dense array of this same size is also allocated for comparison
SPARSE_N, SPARSE_P, SPARSE_T = 300, 1000, 20  # arbitrary


def _check_fitted_tpca_close(tpca1, tpca2, rtol, atol):
    """Check all of the fitted attributes of the two tpca class instances are the 'same'
    """
    assert_allclose(tpca1.n_, tpca2.n_, rtol=rtol, atol=atol)
    assert_allclose(tpca1.p_, tpca2.p_, rtol=rtol, atol=atol)
    assert_allclose(tpca1.t_, tpca2.t_, rtol=rtol, atol=atol)
    assert_allclose(tpca1.k_, tpca2.k_, rtol=rtol, atol=atol)
    assert_allclose(tpca1.n_components_, tpca2.n_components_, rtol=rtol, atol=atol)
    assert_allclose(
        tpca1.explained_variance_ratio_,
        tpca2.explained_variance_ratio_,
        rtol=rtol,
        atol=atol, 
    )
    assert_allclose(
        tpca1.singular_values_, tpca2.singular_values_, rtol=rtol, atol=atol
    )
    assert_allclose(tpca1.mean_, tpca2.mean_, rtol=rtol, atol=atol)
    assert_allclose(tpca1.rho_, tpca2.rho_, rtol=rtol, atol=atol)


@pytest.mark.parametrize("tensor_size", TENSOR_SIZES)
@pytest.mark.parametrize("element_scale", ELEMENT_SCALES)
@pytest.mark.parametrize("include_negatives", [0, 1])
def test_full_tpca(tensor_size, element_scale, include_negatives):
    rng = np.random.default_rng(seed=GLOBAL_SEED)

    n, p, t = tensor_size
    k = min(n, p)
    tpca = TPCA(n_components=None)

    # tensors of various sizes with uniformly distributed elements
    # within [0, tensor_size) or [-0.5*tensor_size, 0.5*tensor_size)
    X = (
        rng.random(size=tensor_size) * element_scale
        - include_negatives * 0.5 * element_scale
    )

    X_r = tpca.fit(X).transform(X)

    # check the output is intended size
    assert len(X_r.shape) == 2
    assert X_r.shape[0] == n
    assert X_r.shape[1] == k * t

    # check the equivalence of fit.transform and fit_transform
    # allow 1e-10 of absolute tolerance for small elements
    X_r2 = tpca.fit_transform(X)
    assert_allclose(X_r, X_r2, rtol=1e-7, atol=1e-10)

    # test rho
    assert tpca.rho_.sum() == k * t

    # test explained variance ratio
    assert_allclose(tpca.explained_variance_ratio_.sum(), 1.0)

"""
@pytest.mark.parametrize("density", [0.01, 0.1, 0.30])
@pytest.mark.parametrize("n_components", [1, 2, 10])
def test_tpca_sparse(density):
    rng = np.random.default_rng(seed=GLOBAL_SEED)
    pass


""" 
"""
def test_tpca_truncation(tensor_size):
    pass

"""
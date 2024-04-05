from functools import reduce

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import special_ortho_group

from tred import (
    generate_default_m_transform_pair,
    TPCA,
    tsvdm,
    m_product,
    generate_transform_pair_from_matrix,
    generate_dctii_m_transform_pair,
    generate_dstii_m_transform_pair,
)


GLOBAL_SEED = 1
RNG = np.random.default_rng(seed=GLOBAL_SEED)


def _dummy_random_orthogonal_m_transform_generator(t):
    """Let's test a m-transform defined by a random orthogonal matrix too!"""
    M_mat = special_ortho_group.rvs(t, random_state=RNG)
    M, Minv = generate_transform_pair_from_matrix(M_mat)
    return M, Minv


def _dummy_default_transform_generator(t):
    """Return M=None, Minv=None to test default m-transform configuration"""
    return None, None


# various n, p, t sizes
# ensure n > p, p > n inputs are tested
TENSOR_SHAPES = [(4, 3, 2), (5, 7, 6), (2, 2, 6)]

# m transforms to suit the tensor sizes above
TRANSFORM_FAMILY_GENERATORS = [
    _dummy_random_orthogonal_m_transform_generator,
    _dummy_default_transform_generator,
    generate_dctii_m_transform_pair,
    generate_dstii_m_transform_pair,
]

# test tiny, small, medium, and large numbers
ELEMENT_SCALES = [10**i for i in range(-4, 5, 4)]


def _check_fitted_tpca_close(tpca1, tpca2, rtol, atol):
    """Check all of the fitted attributes of the two tpca classes
    NOTE: unused at the moment, but will likely be useful in future tests
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


@pytest.mark.parametrize("tensor_shape", TENSOR_SHAPES)
@pytest.mark.parametrize("element_scale", ELEMENT_SCALES)
@pytest.mark.parametrize("include_negatives", [0, 1])
@pytest.mark.parametrize("transform_generator", TRANSFORM_FAMILY_GENERATORS)
def test_tsvdm(tensor_shape, element_scale, include_negatives, transform_generator):
    rng = np.random.default_rng(seed=GLOBAL_SEED)
    n, p, t = tensor_shape

    # tensors of various sizes with uniformly distributed elements
    # within [0, element_scale) or [-0.5*element_scale, 0.5*element_scale)
    X = (
        rng.random(size=tensor_shape) * element_scale
        - include_negatives * 0.5 * element_scale
    )

    M, Minv = transform_generator(t)
    U, S, V = tsvdm(X, M=M, Minv=Minv)
    Vt = V.transpose(1, 0, 2)

    # make sure that the m-product below has a 'non-None' set of inputs for M and Minv
    if M is None:
        M, Minv = generate_default_m_transform_pair(t)

    def m_product_wrapper(A, B):
        """m-product with fixed M and Minv to use for functools reduce"""
        return m_product(A, B, M=M, Minv=Minv)

    X_reconstruct = reduce(m_product_wrapper, (U, S, Vt))
    assert_allclose(X, X_reconstruct)


@pytest.mark.parametrize("tensor_shape", TENSOR_SHAPES)
@pytest.mark.parametrize("element_scale", ELEMENT_SCALES)
@pytest.mark.parametrize("include_negatives", [0, 1])
@pytest.mark.parametrize("n_components", [None, 1, 6, 0.8])
@pytest.mark.parametrize("centre", [True, False])
@pytest.mark.parametrize("transform_generator", TRANSFORM_FAMILY_GENERATORS)
def test_tpca(
    tensor_shape,
    element_scale,
    include_negatives,
    n_components,
    transform_generator,
    centre,
):
    """Make sure different inputs work, and perform basic sense checks"""
    RTOL = 1e-7
    ATOL = 1e-10

    rng = np.random.default_rng(seed=GLOBAL_SEED)

    n, p, t = tensor_shape
    k = min(n, p)

    M, Minv = transform_generator(t)
    tpca = TPCA(n_components=n_components, M=M, Minv=Minv, centre=centre)

    # tensors of various sizes with uniformly distributed elements
    # within [0, element_scale) or [-0.5*element_scale, 0.5*element_scale)
    X = (
        rng.random(size=tensor_shape) * element_scale
        - include_negatives * 0.5 * element_scale
    )

    X_r = tpca.fit(X).transform(X)

    # check the output is intended size
    assert len(X_r.shape) == 2
    assert X_r.shape[0] == n
    if n_components is None:
        assert X_r.shape[1] == k * t
    elif isinstance(n_components, int):
        assert X_r.shape[1] == n_components

    # check the equivalence of fit.transform and fit_transform
    # allow 1e-10 of absolute tolerance for small elements
    X_r2 = tpca.fit_transform(X)
    assert_allclose(X_r, X_r2, rtol=RTOL, atol=ATOL)

    # test rho
    if n_components is None:
        assert tpca.rho_.sum() == k * t
    elif isinstance(n_components, int):
        assert tpca.rho_.sum() == n_components
    else:
        # just check the sum of rho against internal n_components_
        assert tpca.rho_.sum() == tpca.n_components_

    # test explained variance ratio
    if n_components is None:
        assert_allclose(tpca.explained_variance_ratio_.sum(), 1.0)
    elif isinstance(n_components, float):
        # if n_components specifies the minimum amount of explained variance, check that
        # the truncation achieves this
        assert tpca.explained_variance_ratio_.sum() > n_components

    # test the shape of the loadings matrix
    assert tpca.loadings_matrix_.shape == (tpca.n_components_, tpca.p_)

import numpy as np
import pytest
from scipy.stats import special_ortho_group
from numpy.testing import assert_allclose

from tred import (
    generate_transform_pair_from_matrix,
    generate_dctii_m_transform_pair,
    generate_dstii_m_transform_pair,
)

GLOBAL_SEED = 1

TENSOR_SHAPES = [(4, 3, 2), (5, 7, 6), (2, 2, 6)]
MATRIX_SHAPES = [(4, 6), (7, 3)]

TRANSFORM_FAMILY_GENERATORS = [
    generate_dctii_m_transform_pair,
    generate_dstii_m_transform_pair,
]


@pytest.mark.parametrize("tensor_shape", TENSOR_SHAPES)
def test_generate_transform_pair_from_matrix_for_tensor_target(tensor_shape):
    """Compare implementation with (slower) mathematically clear version"""
    # scaling constant (arbitrary)
    C = 5

    rng = np.random.default_rng(seed=GLOBAL_SEED)
    n, p, t = tensor_shape

    # tensors of various sizes with uniformly distributed elements within [-0.5*C, 0.5*C)
    X = rng.random(size=tensor_shape) * C - 0.5 * C
    M_mat = special_ortho_group.rvs(t)

    # apply across tensor tubes
    hatX_expected = np.zeros(shape=tensor_shape)
    for i in range(n):
        for j in range(p):
            hatX_expected[i, j, :] = M_mat @ X[i, j, :]

    # compare with our optimized implementation
    M, Minv = generate_transform_pair_from_matrix(M_mat)
    assert_allclose(hatX_expected, M(X))

    # test inverse transform working as expected
    assert_allclose(X, Minv(M(X)))


@pytest.mark.parametrize("matrix_shape", MATRIX_SHAPES)
def test_generate_transform_pair_from_matrix_for_matrix_target(matrix_shape):
    """Compare implementation with (slower) mathematically clear version"""
    # scaling constant (arbitrary)
    C = 5

    rng = np.random.default_rng(seed=GLOBAL_SEED)
    k, t = matrix_shape

    # matrix of various sizes with uniformly distributed elements within [-0.5*C, 0.5*C)
    X = rng.random(size=matrix_shape) * C - 0.5 * C
    M_mat = special_ortho_group.rvs(t)

    # apply across matrix rows
    hatX_expected = np.zeros(shape=matrix_shape)
    for i in range(k):
        hatX_expected[i, :] = M_mat @ X[i, :]

    # compare with our optimized implementation
    M, Minv = generate_transform_pair_from_matrix(M_mat)
    assert_allclose(hatX_expected, M(X))

    # test inverse transform working as expected
    assert_allclose(X, Minv(M(X)))


@pytest.mark.parametrize("shape", TENSOR_SHAPES + MATRIX_SHAPES)
@pytest.mark.parametrize("transform_generator", TRANSFORM_FAMILY_GENERATORS)
def test_scipy_fft_wrapper_transforms(shape, transform_generator):
    # scaling constant (arbitrary)
    C = 5

    rng = np.random.default_rng(seed=GLOBAL_SEED)
    t = shape[-1]

    # arrays of various sizes with uniformly distributed elements within [-0.5*C, 0.5*C)
    X = rng.random(size=shape) * C - 0.5 * C
    M, Minv = transform_generator(t)

    # ensure transform does not change shape
    assert M(X).shape == X.shape

    # test inverse transform working as expected
    assert_allclose(X, Minv(M(X)))

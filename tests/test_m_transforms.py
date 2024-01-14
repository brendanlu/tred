import numpy as np
import pytest
from scipy.stats import special_ortho_group
from numpy.testing import assert_allclose

from tred import generate_transform_pair_from_matrix

GLOBAL_SEED = 1

TENSOR_SHAPES = [(4, 3, 2), (5, 7, 6), (2, 2, 6)]
MATRIX_SHAPES = [(4, 6), (7, 3)]


@pytest.mark.parametrize("tensor_shape", TENSOR_SHAPES)
def test_generate_transform_pair_from_matrix_for_tensor(tensor_shape):
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

    M, _ = generate_transform_pair_from_matrix(M_mat)
    assert_allclose(hatX_expected, M(X))


@pytest.mark.parametrize("matrix_shape", MATRIX_SHAPES)
def test_generate_transform_pair_from_matrix_for_matrix(matrix_shape):
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

    M, _ = generate_transform_pair_from_matrix(M_mat)
    assert_allclose(hatX_expected, M(X))

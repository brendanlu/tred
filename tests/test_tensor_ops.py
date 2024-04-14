import numpy as np
import pytest
from numpy.testing import assert_allclose

from tred import facewise_product, _mode_1_unfold, _mode_2_unfold, _mode_3_unfold

GLOBAL_SEED = 1

# various n, p, t sizes
# ensure n > p, p > n inputs are tested
TENSOR_SHAPES = [(4, 3, 2), (5, 7, 6), (2, 2, 6)]


@pytest.mark.parametrize("tensor_shape", TENSOR_SHAPES)
@pytest.mark.parametrize("rectangular_offset", [0, 5])
def test_facewise_product(tensor_shape, rectangular_offset):
    """Compare with mathematically clearer (but less efficient) implementations"""
    # scaling constants (arbitrary)
    SCALE1 = 5
    SCALE2 = 6

    rng = np.random.default_rng(seed=GLOBAL_SEED)
    n, p, t = tensor_shape

    # generate some compatibly sized tensors
    A = rng.random(size=(n, p, t)) * SCALE1 - 0.5 * SCALE1

    B = rng.random(size=(p, n + rectangular_offset, t)) * SCALE2 - 0.5 * SCALE2

    # compute expected results using naive implementations
    fp_expected = np.zeros(shape=(n, n + rectangular_offset, t))
    for i in range(t):
        fp_expected[:, :, i] = A[:, :, i] @ B[:, :, i]

    assert_allclose(fp_expected, facewise_product(A, B))

    C = (
        rng.random(size=(n + rectangular_offset, n + rectangular_offset, t)) * SCALE2
        - 0.5 * SCALE2
    )

    # test facewise product with multiple tensor inputs as well
    fp_expected_cumulative = np.zeros(shape=(n, n + rectangular_offset, t))
    for i in range(t):
        fp_expected_cumulative[:, :, i] = fp_expected[:, :, i] @ C[:, :, i]

    assert_allclose(fp_expected_cumulative, facewise_product(A, B, C))


@pytest.mark.parametrize("view", [True, False])
def test_unfolding_explicit(view):
    """Explicitly test and assert the mode-n unfolding example given in:
    Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications. SIAM
    review, 51(3), pp.455-500.
    """
    X = np.array(
        [
            [[1, 13], [4, 16], [7, 19], [10, 22]],
            [[2, 14], [5, 17], [8, 20], [11, 23]],
            [[3, 15], [6, 18], [9, 21], [12, 24]],
        ]
    )

    X_m1_expected = np.array(
        [
            [1, 4, 7, 10, 13, 16, 19, 22],
            [2, 5, 8, 11, 14, 17, 20, 23],
            [3, 6, 9, 12, 15, 18, 21, 24],
        ]
    )

    X_m2_expected = np.array(
        [
            [1, 2, 3, 13, 14, 15],
            [4, 5, 6, 16, 17, 18],
            [7, 8, 9, 19, 20, 21],
            [10, 11, 12, 22, 23, 24],
        ]
    )

    X_m3_expected = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        ]
    )

    assert np.array_equal(_mode_1_unfold(X, view=view), X_m1_expected)
    assert np.array_equal(_mode_2_unfold(X, view=view), X_m2_expected)
    assert np.array_equal(_mode_3_unfold(X, view=view), X_m3_expected)

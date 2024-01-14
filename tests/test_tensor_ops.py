import numpy as np
import pytest
from numpy.testing import assert_allclose

from tred import facewise_product, _mode_1_unfold, _mode_2_unfold, _mode_3_unfold

GLOBAL_SEED = 1

# various n, p, t sizes
# ensure n > p, p > n inputs are tested
TENSOR_SHAPES = [(10, 3, 2), (5, 50, 5), (2, 2, 15)]


@pytest.mark.parametrize("tensor_size", TENSOR_SHAPES)
@pytest.mark.parametrize("include_negatives", [0, 1])
@pytest.mark.parametrize("rectangular_offset", [0, 5])
def test_facewise_product(tensor_size, include_negatives, rectangular_offset):
    """Compare with mathematically clearer (but less efficient) implementations"""
    # scaling constants (arbitrary)
    C1 = 5
    C2 = 6

    rng = np.random.default_rng(seed=GLOBAL_SEED)
    n, p, t = tensor_size

    # generate some compatibly sized tensors
    A = rng.random(size=((n, p, t))) * C1 - include_negatives * 0.5 * C1

    B = (
        rng.random(size=((p, n + rectangular_offset, t))) * C2
        - include_negatives * 0.5 * C2
    )

    # compute expected results using naive implementations
    fp_expected = np.zeros(shape=(n, n + rectangular_offset, t))
    for i in range(t):
        fp_expected[:, :, i] = A[:, :, i] @ B[:, :, i]

    assert_allclose(fp_expected, facewise_product(A, B))


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

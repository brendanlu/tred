import numpy as np
import pytest
from numpy.testing import assert_allclose

from tred import facewise_product

GLOBAL_SEED = 1

# various n, p, t sizes
# ensure n > p, p > n inputs are tested
TENSOR_SIZES = [(10, 3, 2), (5, 50, 5), (2, 2, 15)]


@pytest.mark.parametrize("tensor_size", TENSOR_SIZES)
@pytest.mark.parametrize("include_negatives", [0, 1])
@pytest.mark.parametrize("rectangular_offset", [0, 2, 5])
def test_prod_ops(tensor_size, include_negatives, rectangular_offset):
    """Compare with mathematically clearer (but less efficient) implementations"""
    # scaling constants (arbitrary)
    C1 = 5
    C2 = 6

    rng = np.random.default_rng(seed=GLOBAL_SEED)
    n, p, t = tensor_size

    # tensors of various sizes with uniformly distributed elements
    # within [0, tensor_size) or [-0.5*tensor_size, 0.5*tensor_size)
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

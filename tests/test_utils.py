import numpy as np
import pytest

from tred import _singular_vals_tensor_to_mat, _singular_vals_mat_to_tensor

GLOBAL_SEED = 1

TENSOR_SHAPES = [(6, 4, 2), (5, 7, 6), (4, 4, 6)]


@pytest.mark.parametrize("tensor_shape", TENSOR_SHAPES)
def test_singular_value_compression(tensor_shape):
    rng = np.random.default_rng(seed=GLOBAL_SEED)
    n, p, t = tensor_shape
    k = min(n, p)

    S_mat = rng.integers(-5, 5, size=(k, t))
    S_tens = _singular_vals_mat_to_tensor(S_mat, n, p, t)

    assert np.array_equal(S_mat, _singular_vals_tensor_to_mat(S_tens))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tred import TPCA


# various n, p, t sizes
# ensure n > p, p > n inputs are tested
TENSOR_SIZES = [(100, 30, 20), (20, 1000, 5)]

# test tiny, small, medium, and large numbers
ELEMENT_SCALES = [10**i for i in range(-2, 4)]


@pytest.mark.parametrize("tensor_size", TENSOR_SIZES)
@pytest.mark.parametrize("element_scale", ELEMENT_SCALES)
@pytest.mark.parametrize("include_negatives", [1, 0])
def test_tpca(tensor_size, element_scale, include_negatives):
    # used to generate some test tensors
    RNG = np.random.default_rng(seed=1)

    n, p, t = tensor_size
    k = min(n, p)
    tpca = TPCA(n_components=None)

    # tensors of various sizes with uniformly distributed elements
    # within [0, tensor_size) or [-0.5*tensor_size, 0.5*tensor_size) 
    X = (
        RNG.random(size=tensor_size) * element_scale
        - include_negatives * 0.5 * element_scale
    )

    X_r = tpca.fit(X).transform(X)

    # check the output is intended size
    assert len(X_r.shape) == 2
    assert X_r.shape[0] == n
    assert X_r.shape[1] == k * t

    # check the equivalence of fit.transform and fit_transform
    X_r2 = tpca.fit_transform(X)
    assert_allclose(X_r, X_r2, rtol=1e-7, atol=1e-13)

    # test rho
    assert tpca.rho_.sum() == k * t

    # test explained variance ratio
    assert_allclose(tpca.explained_variance_ratio_.sum(), 1.0)

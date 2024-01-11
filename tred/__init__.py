"""
The `tred` module implements tensor decomposition methods such as tensor PCA and tensor 
SVD. Most of the functionality in this module can be regarded as dimensionality reduction 
strategies for order-3 datasets of size n, p, t, where p >> n > t. 
"""

from ._m_transforms import generate_DCTii_M_transform_pair
from ._tensor_ops import facewise_product, m_product
from ._tensor_pca import tsvdm, TPCA
from ._utils import display_tensor_facewise


# 'private' ------------------------------------------------------------------------------
# We let users access these utilities and operations if they so wish. In the future, we
# may mark many of these as public
from ._utils import _singular_vals_mat_to_tensor, _singular_vals_tensor_to_mat
from ._tensor_ops import _mode0_unfold


__all__ = [
    "generate_DCTii_M_transform_pair",
    "facewise_product",
    "m_product",
    "tsvdm",
    "TPCA",
    "display_tensor_facewise",
]

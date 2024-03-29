"""
The `tred` module implements tensor decomposition methods such as tensor PCA and tensor 
SVD. Most of the functionality in this module can be regarded as dimensionality reduction 
strategies for order-3 datasets of size n, p, t, where p >> n > t. 
"""

from ._m_transforms import (
    generate_transform_pair_from_matrix,
    generate_dstii_m_transform_pair,
    generate_dctii_m_transform_pair,
)
from ._tensor_ops import facewise_product, m_product
from ._tensor_pca import generate_default_m_transform_pair, tsvdm, TPCA
from ._utils import display_tensor_facewise

# private - for testing
##########################################################################################
from ._tensor_ops import _mode_1_unfold, _mode_2_unfold, _mode_3_unfold
from ._utils import _singular_vals_tensor_to_mat, _singular_vals_mat_to_tensor

__all__ = [
    "generate_transform_pair_from_matrix",
    "generate_dctii_m_transform_pair",
    "generate_dstii_m_transform_pair",
    "facewise_product",
    "m_product",
    "generate_default_m_transform_pair",
    "tsvdm",
    "TPCA",
    "display_tensor_facewise",
]

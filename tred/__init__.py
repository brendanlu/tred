"""
The `tred` module implements tensor decomposition methods such as tensor PCA and tensor 
SVD. Most of the functionality in this module can be regarded as dimensionality reduction 
strategies for order-3 datasets of size n, p, t, where p >> n > t. 
"""

from ._m_transforms import (
    generate_dstii_m_transform_pair,
    generate_dctii_m_transform_pair,
)
from ._tensor_ops import facewise_product, m_product
from ._tensor_pca import tsvdm, TPCA
from ._utils import display_tensor_facewise

__all__ = [
    "generate_dctii_m_transform_pair",
    "generate_dstii_m_transform_pair",
    "facewise_product",
    "m_product",
    "tsvdm",
    "TPCA",
    "display_tensor_facewise",
]

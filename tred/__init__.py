"""
The `tred` module implements tensor decomposition methods such as tensor PCA 
and tensor SVD. Most of the functionality in this module can be regarded as 
dimensionality reduction strategies for order-3 datasets of shape n, p, t, 
where p >> n > t. 
"""
__version__ = "0.1.3"


from ._m_transforms import (
    generate_default_m_transform_pair,
    generate_transform_pair_from_matrix,
    generate_dstii_m_transform_pair,
    generate_dctii_m_transform_pair,
)
from ._tensor_ops import facewise_product, m_product, tsvdm
from ._tensor_pca import TPCA
from ._utils import display_tensor_facewise

__all__ = [
    "tsvdm",
    "TPCA",
    "display_tensor_facewise",
    "generate_default_m_transform_pair",
    "generate_transform_pair_from_matrix",
    "generate_dctii_m_transform_pair",
    "generate_dstii_m_transform_pair",
    "facewise_product",
    "m_product",
    "datasets",
]

# private - for testing
###############################################################################
from ._tensor_ops import _mode_1_unfold, _mode_2_unfold, _mode_3_unfold
from ._utils import _singular_vals_tensor_to_mat, _singular_vals_mat_to_tensor


# documentation configurations
###############################################################################
__docformat__ = "numpy"

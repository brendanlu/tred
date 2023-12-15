"""
The :mod:`tred` module implements tensor decomposition methods such as tensor PCA and 
tensor SVD. Most of the functionality in this module can be regarded as dimensionality
reduction strategies for order-3 datasets of size n, p, t, where p >> n > t. 
"""

from ._tensor_pca import tsvdm, TPCA
from ._utils import display_tensor_facewise
from ._m_transforms import generate_DCTii_M_transform_pair

# 'private' ------------------------------------------------------------------------------
# We let users access these utilities and operations if they so wish. In the future, we
# may mark many of these as public
from ._utils import _singular_vals_mat_to_tensor, _singular_vals_tensor_to_mat
from ._tensor_ops import _m_product, _mode0_unfold

"""
In development...

The :mod:`tdim` includes relatively new tensor decomposition methods. Most of the 
algorithms in this module can be regarded as dimensionality reduction techniques, for 
order-3 data. 
"""

from ._tensor_pca import tsvdm, TPCA
from ._utils import display_tensor_facewise
from ._m_transforms import generate_DCTii_M_transform_pair

# for testing ----------------------------------------------------------------------------
from ._utils import _singular_vals_mat_to_tensor, _singular_vals_tensor_to_mat
from ._ops import _m_product, _mode0_unfold

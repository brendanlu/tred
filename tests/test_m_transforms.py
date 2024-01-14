import numpy as np
import pytest

from tred import generate_transform_pair_from_matrix

TENSOR_SHAPES = [(10, 3, 2), (5, 50, 5), (2, 2, 15)]

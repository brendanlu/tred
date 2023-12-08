"""
Much of this package will be inspired by https://github.com/UriaMorP/mprod_package

For the underlying tensor-product framework and the generalization of svd, see [1]. 

For the explicit rank truncation, and the TCAM algorithm, see [2]. 

References: 
[1] Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor algebra for 
optimal representation and compression of multiway data. Proceedings of the National 
Academy of Sciences, 118(28), p.e2015851118.

[2] Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron, H., 2022. 
Dimensionality reduction of longitudinal’omics data using modern tensor factorizations. 
PLoS Computational Biology, 18(7), p.e1010212.

NOTE: The original authors use m, p, n as the dimensions of the tensors, whereas we use
n, p, t instead. 
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import dct, idct


def _default_transform(t):
    """Wrapper around scipy fft to generate functions to perform $\times_3 M$ operations 
    on order-3 tensors, where M is defined by the (scaled) Discrete Cosine Transform. 
    
    See [1] and [2].

    Parameters
    ----------
        t: int
            The length of the transform

    Returns
    -------
        fun_m: Callable[[ArrayLike], ArrayLike]
            A function which expects an order-3 tensor as input, and applies DCT-II to 
            each of the tubal fibres. This preserves the dimensions of the tensor. 

        inv_m: Callable[[ArrayLike], ArrayLike]
            A tensor transform (the inverse of `fun_m`)
    """

    def M(x):
        assert len(x.shape) == 3, "Expecting order 3 tensor input"
        assert x.shape[-1] == t, f"Expecting input shape to be ?x?x{t}"

        return dct(x, type=2, n=t, axis=-1, norm="ortho")

    def Minv(x):
        assert len(x.shape) == 3, "Expecting order 3 tensor input"
        assert x.shape[-1] == t, f"Expecting input shape to be ?x?x{t}"

        return idct(x, type=2, n=t, axis=-1, norm="ortho")

    return M, Minv


def svdm(A, M, Minv, *, hats = False):
    pass

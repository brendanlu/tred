"""Base module"""


from numbers import Real
import numpy as np


class RealNotInt(Real):
    """A type that represents reals that are not instances of int.

    Behaves like float, but also works with values extracted from numpy arrays.
    isintance(1, RealNotInt) -> False
    isinstance(1.0, RealNotInt) -> True

    From sklearn/utils/_param_validation.py
    """


RealNotInt.register(float)


def _facewise_product(A, B):
    """Facewise product s.t. $C_{:,:,i} = A_{:,:,i} B{:,:,i}$

    This could be optimized later if needed.

    Parameters
    ----------
    A : ArrayLike
        $a \times b \times d$ tensor representation

    B : ArrayLike
        $b \times c \times d$ tensor representation

    Returns
    -------
    C : ndarray, shape: (a, c, d)
        facewise tensor product

    """

    # return np.einsum('mpi,pli->mli', A, B)
    # the following is a quicker version of the above using numpy broadcasting
    return np.matmul(A.transpose(2, 0, 1), B.transpose(2, 0, 1)).transpose(1, 2, 0)


def m_product(A, B, M, Minv):
    """Kilmer et al. (2021) tensor m-product for order-3 tensors. See [1]

    Parameters
    ----------

    Returns
    -------
    """
    assert (
        A.shape[1] == B.shape[0] and A.shape[2] == B.shape[2]
    ), "Non conforming dimensions"

    hatA, hatB = M(A), M(B)
    return Minv(_facewise_product(hatA, hatB))

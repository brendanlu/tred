"""General utility functions"""


import numpy as np


def display_tensor_facewise(tens):
    """Numpy - by default prints order-3 arrays as vertical stacks of order-2 arrays, in
    line with their broadcasting rules. This function prints a transpose view so the
    print output is a more intuitive sequence of frontal slices. 

    We use this in notebooks when exploring data.

    Examples
    --------
    >>> import numpy as np
    >>> from tdim import display_tensor_facewise as disp
    >>> test = np.eye(3)[:, :, None] # a 3x3x1 tensor
    >>> print(test) # Numpy ndarrays default __str__ method is not intuitive here
    [[[1.]
    [0.]
    [0.]]

    [[0.]
    [1.]
    [0.]]

    [[0.]
    [0.]
    [1.]]]
    >>> disp(test)
    [[[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]]
    """
    assert len(tens.shape) == 3, "expecting tensor (order-3) array input!"
    print(tens.transpose(2, 0, 1))


def _singular_vals_mat_to_tensor(mat, n, p, t):
    """Decompress $k \times t$ matrix of singular values into $\hat{S}$ from literature.

    Parameters
    ----------
        mat : ArrayLike, shape: (k, t_)
            $k \times t_$ matrix representation, the function checks if `t_ == t`

        n : int
            first dimension of output tensor

        p : int
            second dimension of output tensor

        t : int
            third dimensions of output tensor

    Returns
    -------
        hatS : ndarray, shape: (n, p, t)
            $n \times p \times t$ tensor representation
    """
    k = min(n, p)
    assert k == mat.shape[0] and t == mat.shape[1], "Ensure conforming dimensions"

    hatS = np.zeros((n, p, t))
    for i in range(t):
        hatS[:k, :k, i] = np.diag(mat[:, i])

    return hatS


def _singular_vals_tensor_to_mat(tensor):
    """Compresses $\hat{S}$, a $n \times p \times t$ f-diagonal tensor of singular values
    into a $k \times t$ matrix. Each row contains the $k$ singular values from that
    longitudinal slice.

    Reverses _singular_vals_mat_to_tensor

    Parameters
    ----------
        tensor : ArrayLike, shape: (n, p, t)
            $n \times p \times t$ tensor representation

    Returns
    -------
        mat : ndarray, shape: (min(n, p), t)
            $k \times t$ matrix representation of singular values, where
            $k = \min{(n,p)}$
    """
    return np.diagonal(tensor, axis1=0, axis2=1).transpose()

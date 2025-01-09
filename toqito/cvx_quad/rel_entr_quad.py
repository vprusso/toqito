import numpy as np


def rel_entr_quad(x: np.ndarray, y: np.ndarray, m: int = 3, k: int = 3) -> np.ndarray:
    r"""
    Returns x.*log(x./y) where x and y are vectors of positive numbers.

    This function implements a second-order cone approximation for the relative
    entropy function. Parameters m and k control the accuracy of this approximation:
    m is the number of quadrature nodes to use and k the number of square-roots to take.
    Default (m, k) = (3, 3).

    Examples
    ==========
    Later

    References
    ==========


    x: np.ndarray
      The first input vector of positive numbers.

    y: np.ndarray
      The second input vector of positive numbers.

    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    """
    sx = np.shape(x)
    sy = np.shape(y)
    xs = all([dim == 1 for dim in sx])
    ys = all([dim == 1 for dim in sy])

    if xs:
        z = np.broadcast_to(x, sy)
    elif ys:
        z = np.broadcast_to(y, sx)
    elif sx == sy:
        z = (x, y)
    else:
        raise Exception("Dimensions of x and y are not compatible")

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if not np.isreal(x).all or not np.isreal(y).all:
            raise Exception("x and y must be real")
        t1 = (x < 0) | (y <= 0)
        t2 = (x == 0) & (y >= 0)
        realmin = np.finfo(float).tiny
        x = np.maximum(x, realmin)
        y = np.maximum(y, realmin)
        z = x * np.log(x / y)

        z[t1] = np.inf
        z[t2] = 0
        return z

    elif x.is_constant() or y.is_constant():
        pass
    elif x.is_affine() or y.is_affine():
        pass

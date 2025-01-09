import numpy as np


def random_psd(dim: int, is_cplx: bool) -> np.ndarray:
    r"""
    Returns a random positive semidefinite matrix of size n.

    Examples
    ==========
    Later

    References
    ==========
    .. bibliography::
           :filter: docname in docnames

    dim: int
      The number of rows (and columns) in the matrix

    iscplx: bool
      Whether to

    """
    A = np.random.randn(dim, dim)
    if is_cplx:
        A = A + 1j * np.random.randn(dim, dim)
    X = np.dot(A, A.conjugate().T)
    return X

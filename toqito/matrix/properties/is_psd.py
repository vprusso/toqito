import numpy as np


def is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is positive semidefinite.
    :param mat: Matrix to check.
    :param tol: Tolerance for numerical accuracy.
    :return: Return True if matrix is PSD and False otherwise.
    """
    ret_mat = np.linalg.eigvalsh(mat)
    return np.all(ret_mat > -tol)


"""kp-norm for matrices."""
import numpy as np


def kp_norm(mat: np.ndarray, k: int, p: int) -> float:
    """
    Compute the p-norm of the vector or the k-largest singular values of a matrix.


    :param mat: 2D numpy ndarray
    :param k: The number of singular values to take.
    :param p: The order of the norm.
    :return: The kp-norm of a matrix.
    """
    dim = min(mat.shape)

    # If the requested norm is the Frobenius norm, compute it using numpy's
    # built-in Frobenius norm calculation, which is significantly faster than
    # computing singular values.
    if k >= dim and p == 2:
        return np.linalg.norm(mat, ord="fro")

    s_vals = np.linalg.svd(mat, compute_uv=False)
    return np.linalg.norm(s_vals[:k], ord=p)

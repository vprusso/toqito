"""Checks whether operator is in the ball of separability centered at the maximally-mixed state."""
import numpy as np


def in_separable_ball(mat: np.ndarray) -> bool:
    r"""
    Check whether an operator is contained in ball of separability [GB02]_.

    Determines whether :code:`mat` is contained within the ball of separable operators centered
    at the identity matrix (i.e. the maximally-mixed state). The size of this ball was derived in
    [GB02]_.

    This function can be used as a method for separability testing of states in certain scenarios.

    This function is adapted from QETLAB.

    Examples
    ==========

    The only states acting on :math:`\complex^m \otimes \complex^n` in the separable ball that do
    not have full rank are those with exactly 1 zero eigenvalue, and the :math:`mn - 1` non-zero
    eigenvalues equal to each other.

    The following is an example of generating a random density matrix with eigenvalues
    :code:`[1, 1, 1, 0]/3`. This example yields a matrix that is contained within the separable
    ball.

    >>> from toqito.random import random_unitary
    >>> from toqito.state_props import in_separable_ball
    >>> import numpy as np
    >>>
    >>> U = random_unitary(4)
    >>> lam = np.array([1, 1, 1, 0]) / 3
    >>> rho = U @ np.diag(lam) @ U.conj().T
    >>> in_separable_ball(rho)
    True

    The following is an example of generating a random density matrix with eigenvalues
    :code:`[1.01, 1, 0.99, 0]/3`. This example yields a matrix that is not contained within the
    separable ball.

    >>> from toqito.random import random_unitary
    >>> from toqito.state_props import in_separable_ball
    >>> import numpy as np
    >>>
    >>> U = random_unitary(4)
    >>> lam = np.array([1.01, 1, 0,.99, 0]) / 3
    >>> rho = U @ np.diag(lam) @ U.conj().T
    >>> in_separable_ball(rho)
    False

    References
    ==========
    .. [GB02] Gurvits, Leonid, and Barnum, Howard.
        "Largest separable balls around the maximally mixed bipartite quantum state."
        Physical Review A 66.6 (2002): 062311.
        https://arxiv.org/pdf/quant-ph/0204159.pdf

    :param mat: A positive semidefinite matrix or a vector of the eigenvalues of a positive
                semidefinite matrix.
    :return: :code:`True` if the matrix :code:`mat` is contained within the separable ball, and
            :code:`False` otherwise.
    """
    mat_dims = mat.shape
    max_dim = max(mat_dims)

    # If the matrix is a vector, turn it into a matrix. We could instead turn every matrix into a
    # vector of eigenvalues, but that would make the computation take O(n^3) time instead of the
    # current method which is O(n^2).

    # Case: Vector of eigenvalues.
    if len(mat_dims) == 1 or min(mat_dims) == 1:
        mat = np.diag(mat)

    # If the matrix has trace equal to 0 or less, it cannot be in the separable ball.
    if np.trace(mat) < max_dim * np.finfo(float).eps:
        return False

    mat = mat / np.trace(mat)

    # The following check relies on the fact that we scaled the matrix so that trace(mat) = 1.
    # The following condition is then exactly the condition mentioned in [GB02]_.
    return np.linalg.norm(mat / np.linalg.norm(mat, "fro") ** 2 - np.eye(max_dim), "fro") <= 1

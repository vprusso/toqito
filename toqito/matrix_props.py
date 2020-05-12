"""Properties of matrices and vectors."""
import numpy as np


__all__ = [
    "is_commuting",
    "is_density",
    "is_diagonal",
    "is_hermitian",
    "is_normal",
    "is_pd",
    "is_projection",
    "is_psd",
    "is_square",
    "is_symmetric",
    "is_unitary",
]


def is_commuting(mat_1: np.ndarray, mat_2: np.ndarray) -> bool:
    r"""
    Determine if two linear operators commute with each other [WikCom]_.

    For any pair of operators :math:`X, Y \in \text{L}(\mathcal{X})`, the
    Lie bracket :math:`\left[X, Y\right] \in \text{L}(\mathcal{X})` is defined
    as

    .. math::
        \left[X, Y\right] = XY - YX.

    It holds that :math:`` if and only if :math:`X` and :math:`Y` commute
    [WatCom18]_.

    Examples
    ==========

    Consider the following matrices:

    .. math::
        A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},
        \quad \text{and} \quad
        B = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.

    It holds that :math:`AB=0`, however

    .. math::
        BA = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} = A,

    and hence, do not commute.

    >>> from toqito.matrix_props import is_commuting
    >>> import numpy as np
    >>> mat_1 = np.array([[0, 1], [0, 0]])
    >>> mat_2 = np.array([[1, 0], [0, 0]])
    >>> is_commuting(mat_1, mat_2)
    False

    Consider the following pair of matrices

    .. math::
        A = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            1 & 0 & 2
            \end{pmatrix}, \quad \text{and} \quad
        B = \begin{pmatrix}
            2 & 4 & 0 \\
            3 & 1 & 0 \\
            -1 & -4 & 1
            \end{pmatrix}.

    It may be verified that :math:`AB = BA = 0`, and therefore :math`A` and
    :math:`B` commute.

    >>> from toqito.matrix_props import is_commuting
    >>> import numpy as np
    >>> mat_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    >>> mat_2 = np.array([[2, 4, 0], [3, 1, 0], [-1, -4, 1]])
    >>> is_commuting(mat_1, mat_2)
    True

    References
    ==========
    .. [WikCom] Wikipedia: Commuting matrices
        https://en.wikipedia.org/wiki/Commuting_matrices

    .. [WatCom18] Watrous, John.
        "The theory of quantum information."
        Section: "Lie brackets and commutants".
        Cambridge University Press, 2018.

    :param mat_1: First matrix to check.
    :param mat_2: Second matrix to check.
    :return: Return `True` if `mat_1` commutes with `mat_2` and False otherwise.
    """
    return np.allclose(np.matmul(mat_1, mat_2) - np.matmul(mat_2, mat_1), 0)


def is_density(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is a density matrix [WikDensity]_.

    A matrix is a density matrix if its trace is equal to one and it has the
    property of being positive semidefinite (PSD).

    Examples
    ==========

    Consider the Bell state

    .. math::
       u = \frac{1}{\sqrt{2}} |00 \rangle + \frac{1}{\sqrt{2}} |11 \rangle.

    Constructing the matrix :math:`\rho = u u^*` defined as

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                                1 & 0 & 0 & 1 \\
                                0 & 0 & 0 & 0 \\
                                0 & 0 & 0 & 0 \\
                                1 & 0 & 0 & 1
                           \end{pmatrix}

    our function indicates that this is indeed a density operator as the trace
    of :math:`\rho` is equal to :math:`1` and the matrix is positive
    semidefinite:

    >>> from toqito.matrix_props import is_density
    >>> from toqito.states import bell
    >>> import numpy as np
    >>> rho = bell(0) * bell(0).conj().T
    >>> is_density(rho)
    True

    Alternatively, the following example matrix :math:`\sigma` defined as

    .. math::
        \sigma = \frac{1}{2} \begin{pmatrix}
                                1 & 2 \\
                                3 & 1
                             \end{pmatrix}

    does satisfy :math:`\text{Tr}(\sigma) = 1`, however fails to be positive
    semidefinite, and is therefore not a density operator. This can be
    illustrated using `toqito` as follows.

    >>> from toqito.matrix_props import is_density
    >>> from toqito.states import bell
    >>> import numpy as np
    >>> sigma = 1/2 * np.array([[1, 2], [3, 1]])
    >>> is_density(sigma)
    False

    References
    ==========
    .. [WikDensity] Wikipedia: Density matrix
        https://en.wikipedia.org/wiki/Density_matrix

    :param mat: Matrix to check.
    :return: Return `True` if matrix is a density matrix, and `False`
             otherwise.
    """
    return is_psd(mat) and np.isclose(np.trace(mat), 1)


def is_diagonal(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is diagonal [WikDiag]_.

    A matrix is diagonal if the matrix is square and if the diagonal of the
    matrix is non-zero, while the off-diagonal elements are all zero.

    The following is an example of a 3-by-3 diagonal matrix:

    .. math::
        \begin{equation}
            \begin{pmatrix}
                1 & 0 & 0 \\
                0 & 2 & 0 \\
                0 & 0 & 3
            \end{pmatrix}
        \end{equation}

    This quick implementation is given by Daniel F. from StackOverflow in
    [SODIA]_.

    Examples
    ==========

    Consider the following diagonal matrix

    .. math::
        A = \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}.

    Our function indicates that this is indeed a diagonal matrix:

    >>> from toqito.matrix_props import is_diagonal
    >>> import numpy as np
    >>> A = np.array([[1, 0], [0, 1]])
    >>> is_diagonal(A)
    True

    Alternatively, the following example matrix

    .. math::
        B = \begin{pmatrix}
                1 & 2 \\
                3 & 4
            \end{pmatrix}

    is not diagonal, as shown using `toqito`

    >>> from toqito.matrix_props import is_diagonal
    >>> import numpy as np
    >>> B = np.array([[1, 2], [3, 4]])
    >>> is_diagonal(B)
    False

    References
    ==========
    .. [WikDiag] Wikipedia: Diagonal matrix
        https://en.wikipedia.org/wiki/Diagonal_matrix

    .. [SODIA] StackOverflow post
        https://stackoverflow.com/questions/43884189/

    :param mat: The matrix to check.
    :return: Returns True if the matrix is diagonal and False otherwise.
    """
    if not is_square(mat):
        return False
    i, j = mat.shape
    test = mat.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


def is_hermitian(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is Hermitian [WikHerm]_.

    A Hermitian matrix is a complex square matrix that is equal to its own
    conjugate transpose.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                2 & 2 +1j & 4 \\
                2 - 1j & 3 & 1j \\
                4 & -1j & 1
            \end{pmatrix}

    our function indicates that this is indeed a Hermitian matrix as it holds that

    .. math::
        A = A^*.

    >>> from toqito.matrix_props import is_hermitian
    >>> import numpy as np
    >>> mat = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])
    >>> is_hermitian(mat)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    is not Hermitian.

    >>> from toqito.matrix_props import is_hermitian
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_hermitian(mat)
    False

    References
    ==========
    .. [WikHerm] Wikipedia: Hermitian matrix.
        https://en.wikipedia.org/wiki/Hermitian_matrix

    :param mat: Matrix to check.
    :return: Return True if matrix is Hermitian, and False otherwise.
    """
    return np.allclose(mat, mat.conj().T)


def is_normal(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Determine if a matrix is normal [WikNormal]_.

    A matrix is normal if it commutes with its adjoint

    .. math::
        \begin{equation}
            [X, X^*] = 0,
        \end{equation}

    or, equivalently if

    .. math::
        \begin{equation}
            X^* X = X X^*.
        \end{equation}

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    our function indicates that this is indeed a normal matrix.

    >>> from toqito.matrix_props import is_normal
    >>> import numpy as np
    >>> A = np.identity(4)
    >>> is_normal(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    is not normal.

    >>> from toqito.matrix_props import is_normal
    >>> import numpy as np
    >>> B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_normal(B)
    False

    References
    ==========
    .. [WikNormal] Wikipedia: Normal matrix.
        https://en.wikipedia.org/wiki/Normal_matrix

    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Returns True if the matrix is normal and False otherwise.
    """
    return np.allclose(
        np.matmul(mat, mat.conj().T), np.matmul(mat.conj().T, mat), rtol=rtol, atol=atol
    )


def is_pd(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is positive definite (PD) [WikPD]_.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                2 & -1 & 0 \\
                -1 & 2 & -1 \\
                0 & -1 & 2
            \end{pmatrix}

    our function indicates that this is indeed a positive definite matrix.

    >>> from toqito.matrix_props import is_pd
    >>> import numpy as np
    >>> A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    >>> is_pd(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive definite.

    >>> from toqito.matrix_props import is_pd
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_pd(B)
    False

    See Also
    ========
    is_psd

    References
    ==========
    .. [WikPD] Wikipedia: Definiteness of a matrix.
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix

    :param mat: Matrix to check.
    :return: Return True if matrix is PD, and False otherwise.
    """
    if np.array_equal(mat, mat.T):
        try:
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def is_projection(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is a projection matrix [WikProj]_.

    A matrix is a projection matrix if it is positive semidefinite (PSD) and if

    .. math::
        \begin{equation}
            X^2 = X
        \end{equation}

    where :math:`X` is the matrix in question.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                0 & 1 \\
                0 & 1
            \end{pmatrix}

    our function indicates that this is indeed a projection matrix.

    >>> from toqito.matrix_props import is_projection
    >>> import numpy as np
    >>> A = np.array([[0, 1], [0, 1]])
    >>> is_projection(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive definite.

    >>> from toqito.matrix_props import is_projection
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_projection(B)
    False

    References
    ==========
    .. [WikProj] Wikipedia: Projection matrix.
        https://en.wikipedia.org/wiki/Projection_matrix

    :param mat: Matrix to check.
    :return: Return True if matrix is a projection matrix, and False otherwise.
    """
    if not is_psd(mat):
        return False
    return np.allclose(np.linalg.matrix_power(mat, 2), mat)


def is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    r"""
    Check if matrix is positive semidefinite (PSD) [WikPSD]_.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & -1 \\
                -1 & 1
            \end{pmatrix}

    our function indicates that this is indeed a positive semidefinite matrix.

    >>> from toqito.matrix_props import is_psd
    >>> import numpy as np
    >>> A = np.array([[1, -1], [-1, 1]])
    >>> is_psd(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive semidefinite.

    >>> from toqito.matrix_props import is_psd
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_psd(B)
    False

    References
    ==========
    .. [WikPSD] Wikipedia: Definiteness of a matrix.
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix

    :param mat: Matrix to check.
    :param tol: Tolerance for numerical accuracy.
    :return: Return True if matrix is PSD, and False otherwise.
    """
    if not is_square(mat):
        return False
    return np.all(np.linalg.eigvalsh(mat) > -tol)


def is_square(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is square [WikSquare]_.

    A matrix is square if the dimensions of the rows and columns are equivalent.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    our function indicates that this is indeed a square matrix.

    >>> from toqito.matrix_props import is_square
    >>> import numpy as np
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_square(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6
            \end{pmatrix}

    is not square.

    >>> from toqito.matrix_props import is_square
    >>> import numpy as np
    >>> B = np.array([[1, 2, 3], [4, 5, 6]])
    >>> is_square(B)
    False

    References
    ==========
    .. [WikSquare] Wikipedia: Square matrix.
        https://en.wikipedia.org/wiki/Square_matrix

    :param mat: The matrix to check.
    :return: Returns True if the matrix is square and False otherwise.
    """
    return mat.shape[0] == mat.shape[1]


def is_symmetric(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Determine if a matrix is symmetric [WikSym]_.

    The following 3x3 matrix is an example of a symmetric matrix:

    .. math::

        \begin{pmatrix}
            1 & 7 & 3 \\
            7 & 4 & -5 \\
            3 &-5 & 6
        \end{pmatrix}

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 7 & 3 \\
                7 & 4 & -5 \\
                3 & -5 & 6
            \end{pmatrix}

    our function indicates that this is indeed a symmetric matrix.

    >>> from toqito.matrix_props import is_symmetric
    >>> import numpy as np
    >>> A = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])
    >>> is_symmetric(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 \\
                4 & 5
            \end{pmatrix}

    is not symmetric.

    >>> from toqito.matrix_props import is_symmetric
    >>> import numpy as np
    >>> B = np.array([[1, 2], [3, 4]])
    >>> is_symmetric(B)
    False

    References
    ==========
    .. [WikSym] Wikipedia: Symmetric matrix
        https://en.wikipedia.org/wiki/Symmetric_matrix

    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Returns True if the matrix is symmetric and False otherwise.
    """
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)


def is_unitary(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is unitary [WikUnitary]_.

    A matrix is unitary if its inverse is equal to its conjugate transpose.

    Alternatively, a complex square matrix :math:`U` is unitary if its conjugate
    transpose :math:`U^*` is also its inverse, that is, if

    .. math::
        \begin{equation}
            U^* U = U U^* = \mathbb{I},
        \end{equation}

    where :math:`\mathbb{I}` is the identity matrix.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
            0 & 1 \\
            1 & 0
            \end{pmatrix}

    our function indicates that this is indeed a unitary matrix.

    >>> from toqito.matrix_props import is_unitary
    >>> import numpy as np
    >>> A = np.array([[0, 1], [1, 0]])
    >>> is_unitary(A)
    True

    We may also use the `random_unitary` function from `toqito`, and can verify
    that a randomly generated matrix is unitary

    >>> from toqito.matrix_props import is_unitary
    >>> from toqito.random import random_unitary
    >>> mat = random_unitary(2)
    >>> is_unitary(mat)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
            1 & 0 \\
            1 & 1
            \end{pmatrix}

    is not unitary.

    >>> from toqito.matrix_props import is_unitary
    >>> import numpy as np
    >>> B = np.array([[1, 0], [1, 1]])
    >>> is_unitary(B)
    False

    References
    ==========
    .. [WikUnitary] Wikipedia: Unitary matrix.
        https://en.wikipedia.org/wiki/Unitary_matrix

    :param mat: Matrix to check.
    :return: Return `True` if matrix is unitary, and `False` otherwise.
    """
    # If U^* * U = I U * U^*, the matrix "U" is unitary.
    return np.allclose(np.eye(len(mat)), mat.dot(mat.conj().T))

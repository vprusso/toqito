"""Check if a matrix forms an equiangular tight frame (ETF)."""

import numpy as np

from toqito.matrix_props.is_tight_frame import is_tight_frame


def is_equiangular_tight_frame(mat: np.ndarray, tol: float = 1e-8) -> bool:
    r"""Check if the columns of a matrix form an equiangular tight frame (ETF).

    A matrix \(M \in \mathbb{C}^{d \times n}\) forms an equiangular tight frame if:

    1. Each column has unit norm: \(\|m_i\| = 1\) for all \(i\).
    2. The columns are equiangular: \(|\langle m_i, m_j \rangle|\) is constant
       for all \(i \neq j\).
    3. The columns form a tight frame: \(M M^* = \frac{n}{d} I_d\).

    The Welch bound gives the minimum achievable angle:

    \[
        |\langle m_i, m_j \rangle| \geq \sqrt{\frac{n - d}{d(n - 1)}}
    \]

    and ETFs achieve this bound with equality.

    For further details, see [@tropp2005complex].

    Args:
        mat: A 2D numpy array of shape (d, n) whose columns are the frame vectors.
        tol: Numerical tolerance. Default 1e-8.

    Returns:
        `True` if the columns of `mat` form an equiangular tight frame; `False` otherwise.

    Examples:
        The columns of a Hadamard matrix (rescaled) form an ETF.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_equiangular_tight_frame

        # 3 equiangular vectors in R^2 (Mercedes-Benz frame)
        mat = np.array([
            [0, np.sqrt(3)/2, -np.sqrt(3)/2],
            [1, -1/2, -1/2],
        ])
        print(is_equiangular_tight_frame(mat))
        ```

        A random matrix is generally not an ETF.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_equiangular_tight_frame

        mat = np.array([[1, 0], [0, 1], [1, 1]])
        print(is_equiangular_tight_frame(mat))
        ```

    """
    if mat.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    dim, num_vecs = mat.shape

    if num_vecs < 2:
        # A single vector trivially satisfies ETF conditions if it has unit norm.
        return np.isclose(np.linalg.norm(mat[:, 0]), 1.0, atol=tol) if num_vecs == 1 else False

    # 1. Check unit norms.
    norms = np.linalg.norm(mat, axis=0)
    if not np.allclose(norms, 1.0, atol=tol):
        return False

    # 2. Check equiangularity.
    gram = mat.conj().T @ mat
    # Extract absolute values of off-diagonal inner products.
    abs_inner = np.abs(gram[np.triu_indices(num_vecs, k=1)])
    if abs_inner.size > 0 and (np.max(abs_inner) - np.min(abs_inner)) > tol:
        return False

    # 3. Check tight frame.
    vectors = [mat[:, i] for i in range(num_vecs)]
    return is_tight_frame(vectors, tol=tol)

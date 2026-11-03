"""Pyramid state."""

import numpy as np


def pyramid(idx: int) -> np.ndarray:
    r"""Produce a Pyramid state [@bennett1999unextendible].

    The Pyramid states constitute five product states on 3-by-3 dimensional space that form a UPB
    (unextendible product basis). They are built from five unit vectors pointing from the center of a
    regular pentagon to the apex of a pyramid above it,

    \[
        |v_j\rangle = N \left( \cos\frac{2\pi j}{5}, \ \sin\frac{2\pi j}{5}, \ h \right),
        \qquad h = \frac{1}{2}\sqrt{1 + \sqrt{5}}, \qquad j = 0, \dots, 4,
    \]

    where the height \(h\) is chosen so that non-adjacent vectors are orthogonal and \(N\) normalizes
    the vector. The product states are

    \[
        |\psi_j\rangle = |v_j\rangle \otimes |v_{2j \bmod 5}\rangle, \qquad j = 0, \dots, 4.
    \]

    Two members whose indices differ by two are orthogonal on the first factor, and two whose indices
    differ by one are orthogonal on the second.

    Args:
        idx: A parameter in `[0, 1, 2, 3, 4]`.

    Returns:
        Pyramid state.

    Raises:
        ValueError: Invalid value for `idx`.

    Examples:
        The five Pyramid states are orthonormal.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import pyramid
        vecs = np.column_stack([pyramid(i) for i in range(5)])
        print(np.allclose(vecs.T @ vecs, np.identity(5)))
        ```

    """
    if idx not in range(5):
        raise ValueError("Invalid integer value for Pyramid state.")
    height = np.sqrt(1 + np.sqrt(5)) / 2

    def vertex(j: int) -> np.ndarray:
        vec = np.array([[np.cos(2 * np.pi * j / 5)], [np.sin(2 * np.pi * j / 5)], [height]])
        return vec / np.linalg.norm(vec)

    return np.kron(vertex(idx), vertex((2 * idx) % 5))

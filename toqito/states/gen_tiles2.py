"""GenTiles2 unextendible product basis."""

import numpy as np


def gen_tiles2(dim_m: int, dim_n: int) -> list[np.ndarray]:
    r"""Produce the GenTiles2 unextendible product basis [@divincenzo2003unextendible].

    GenTiles2 generalizes the Tile UPB of \(\mathbb{C}^3 \otimes \mathbb{C}^3\) to
    \(\mathbb{C}^m \otimes \mathbb{C}^n\) with \(n > 3\), \(m \geq 3\) and \(n \geq m\). It consists of
    the \(m\) short tiles

    \[
        |S_j\rangle = \frac{1}{\sqrt{2}} \left( |j\rangle - |j + 1 \bmod m\rangle \right) \otimes |j\rangle,
        \qquad 0 \leq j \leq m - 1,
    \]

    the \(m(n - 3)\) long tiles, with \(\omega = e^{2\pi i/(n-2)}\),

    \[
        |L_{j,k}\rangle = |j\rangle \otimes \frac{1}{\sqrt{n - 2}} \left(
            \sum_{i=0}^{m-3} \omega^{ik} \, |i + j + 1 \bmod m\rangle
            + \sum_{i=m-2}^{n-3} \omega^{ik} \, |i + 2\rangle \right),
        \qquad 0 \leq j \leq m - 1, \quad 1 \leq k \leq n - 3,
    \]

    and the stopper \(|F\rangle = \frac{1}{\sqrt{mn}} \sum_{i,j} |i\rangle \otimes |j\rangle\), for a total
    of \(mn - 2m + 1\) orthonormal product vectors. The bound entangled state on their orthogonal
    complement has rank \(2m - 1\) and can be built with `toqito.states.upb_state`.

    Args:
        dim_m: The dimension \(m\) of the first party, at least 3.
        dim_n: The dimension \(n\) of the second party, larger than 3 and at least \(m\).

    Returns:
        A list of the \(mn - 2m + 1\) product vectors as column vectors, short tiles first, then long
        tiles, then the stopper.

    Raises:
        ValueError: If the dimensions do not satisfy \(n > 3\), \(m \geq 3\) and \(n \geq m\).

    Examples:
        On \(\mathbb{C}^3 \otimes \mathbb{C}^4\) the basis has seven members and the associated state has
        rank five.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import gen_tiles2, upb_state
        vecs = gen_tiles2(3, 4)
        print(len(vecs), np.linalg.matrix_rank(upb_state(vecs)))
        ```

    """
    if not (dim_n > 3 and dim_m >= 3 and dim_n >= dim_m):
        raise ValueError("GenTiles2 requires n > 3, m >= 3 and n >= m.")
    eye_m, eye_n = np.identity(dim_m), np.identity(dim_n)
    omega = np.exp(2j * np.pi / (dim_n - 2))

    short = [np.kron((eye_m[:, [j]] - eye_m[:, [(j + 1) % dim_m]]) / np.sqrt(2), eye_n[:, [j]]) for j in range(dim_m)]
    long_tiles = []
    for j in range(dim_m):
        for k in range(1, dim_n - 2):
            vec = np.zeros((dim_n, 1), dtype=complex)
            for i in range(dim_m - 2):
                vec[(i + j + 1) % dim_m] += omega ** (i * k)
            for i in range(dim_m - 2, dim_n - 2):
                vec[i + 2] += omega ** (i * k)
            long_tiles.append(np.kron(eye_m[:, [j]], vec / np.sqrt(dim_n - 2)))
    stopper = np.ones((dim_m * dim_n, 1), dtype=complex) / np.sqrt(dim_m * dim_n)
    return short + long_tiles + [stopper]

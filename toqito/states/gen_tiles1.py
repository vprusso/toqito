"""GenTiles1 unextendible product basis."""

import numpy as np


def gen_tiles1(dim: int) -> list[np.ndarray]:
    r"""Produce the GenTiles1 unextendible product basis [@divincenzo2003unextendible].

    GenTiles1 generalizes the Tile UPB of \(\mathbb{C}^3 \otimes \mathbb{C}^3\) to
    \(\mathbb{C}^d \otimes \mathbb{C}^d\) for every even local dimension \(d \geq 4\). With
    \(\omega = e^{4\pi i/d}\), define the "half-filled" vectors

    \[
        |\omega_{m,k}\rangle = \frac{1}{\sqrt{d/2}} \sum_{j=0}^{d/2 - 1} \omega^{jm} \, |(j + k) \bmod d\rangle.
    \]

    For \(m = 1, \dots, d/2 - 1\) and \(k = 0, \dots, d - 1\) the basis consists of the vertical tiles
    \(|V_{m,k}\rangle = |k\rangle \otimes |\omega_{m,k+1}\rangle\), the horizontal tiles
    \(|H_{m,k}\rangle = |\omega_{m,k}\rangle \otimes |k\rangle\), and the stopper
    \(|F\rangle = \frac{1}{d} \sum_{i,j} |i\rangle \otimes |j\rangle\), for a total of \(d^2 - 2d + 1\)
    orthonormal product vectors. The bound entangled state on their orthogonal complement has rank
    \(2d - 1\) and can be built with `toqito.states.upb_state`.

    Args:
        dim: The local dimension \(d\), an even integer at least 4.

    Returns:
        A list of the \(d^2 - 2d + 1\) product vectors as column vectors, vertical tiles first, then
        horizontal tiles, then the stopper.

    Raises:
        ValueError: If `dim` is odd or smaller than 4.

    Examples:
        For \(d = 4\) the basis has nine members and the associated state has rank seven.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import gen_tiles1, upb_state
        vecs = gen_tiles1(4)
        print(len(vecs), np.linalg.matrix_rank(upb_state(vecs)))
        ```

    """
    if dim < 4 or dim % 2:
        raise ValueError("The local dimension must be an even integer at least 4.")
    half = dim // 2
    omega = np.exp(4j * np.pi / dim)

    def half_filled(m_idx: int, k_idx: int) -> np.ndarray:
        vec = np.zeros((dim, 1), dtype=complex)
        for j in range(half):
            vec[(j + k_idx) % dim] += omega ** (j * m_idx)
        return vec / np.sqrt(half)

    identity = np.identity(dim)
    vertical = [np.kron(identity[:, [k]], half_filled(m_idx, k + 1)) for m_idx in range(1, half) for k in range(dim)]
    horizontal = [np.kron(half_filled(m_idx, k), identity[:, [k]]) for m_idx in range(1, half) for k in range(dim)]
    stopper = np.ones((dim * dim, 1), dtype=complex) / dim
    return vertical + horizontal + [stopper]

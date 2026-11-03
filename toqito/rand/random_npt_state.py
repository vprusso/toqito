"""Generates a random density matrix with negative partial transpose."""

import numpy as np

from toqito.rand._ppt_sampling import _rejection_sample


def random_npt_state(dims: list[int], max_draws: int = 100_000, seed: int | None = None) -> np.ndarray:
    r"""Generate a random bipartite density matrix with negative partial transpose (NPT).

    Density matrices on \(\mathbb{C}^{d_A} \otimes \mathbb{C}^{d_B}\) are drawn from the Hilbert-Schmidt
    measure with `toqito.rand.random_density_matrix` and rejected until one fails the PPT test. The
    result is therefore distributed as the Hilbert-Schmidt measure conditioned on NPT.

    An NPT state is entangled and distillable, so it is never bound entangled. Such draws serve as
    comparison points for the bound entangled states in `toqito.states`, which are all PPT.

    Over Hilbert-Schmidt draws the NPT fraction is about \(0.8\) on \(2 \otimes 2\) and essentially one on
    \(3 \otimes 3\) and larger, so the rejection loop rarely needs more than a few draws. The function
    raises after `max_draws` failed attempts rather than looping indefinitely.

    Args:
        dims: The local dimensions \([d_A, d_B]\).
        max_draws: Maximum number of draws before giving up.
        seed: A seed used to instantiate numpy's random number generator.

    Returns:
        A \(d_A d_B\)-by-\(d_A d_B\) random NPT density matrix.

    Raises:
        ValueError: If `dims` is not a list of two positive integers.
        RuntimeError: If no NPT state is found within `max_draws` draws.

    Examples:
        A random NPT state has a negative eigenvalue after partial transposition.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import partial_transpose
        from toqito.rand import random_npt_state
        rho = random_npt_state([3, 3], seed=42)
        print(np.linalg.eigvalsh(partial_transpose(rho, [1], [3, 3])).min() < 0)
        ```

    """
    gen = np.random.default_rng(seed=seed)
    return _rejection_sample(dims, False, max_draws, gen)

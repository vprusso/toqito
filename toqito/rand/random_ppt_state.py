"""Generates a random density matrix with positive partial transpose."""

import numpy as np

from toqito.rand._ppt_sampling import _rejection_sample


def random_ppt_state(dims: list[int], max_draws: int = 100_000, seed: int | None = None) -> np.ndarray:
    r"""Generate a random bipartite density matrix with positive partial transpose (PPT).

    Density matrices on \(\mathbb{C}^{d_A} \otimes \mathbb{C}^{d_B}\) are drawn from the Hilbert-Schmidt
    measure with `toqito.rand.random_density_matrix` and rejected until one passes the PPT test. The
    result is therefore distributed as the Hilbert-Schmidt measure conditioned on PPT.

    A PPT draw is not a bound entangled state. On \(2 \otimes 2\) and \(2 \otimes 3\) every PPT state
    is separable, and in higher dimensions a random PPT state may well be separable too; a separability
    test such as `toqito.state_props.is_separable` is needed to say more.

    Rejection sampling is only practical when the PPT fraction is not too small. Over Hilbert-Schmidt
    draws the fraction is about \(0.2\) on \(2 \otimes 2\) and about \(0.025\) on \(2 \otimes 3\), while
    on \(3 \otimes 3\) and \(4 \otimes 4\) none of 20000 draws was PPT. The function therefore raises
    after `max_draws` failed attempts rather than looping indefinitely.

    Args:
        dims: The local dimensions \([d_A, d_B]\).
        max_draws: Maximum number of draws before giving up.
        seed: A seed used to instantiate numpy's random number generator.

    Returns:
        A \(d_A d_B\)-by-\(d_A d_B\) random PPT density matrix.

    Raises:
        ValueError: If `dims` is not a list of two positive integers.
        RuntimeError: If no PPT state is found within `max_draws` draws.

    Examples:
        A random two-qubit PPT state is separable, since PPT and separability coincide there.

        ```python exec="1" source="above" result="text"
        from toqito.rand import random_ppt_state
        from toqito.state_props import is_ppt, is_separable
        rho = random_ppt_state([2, 2], seed=42)
        print(is_ppt(rho, dim=[2, 2]), is_separable(rho, dim=[2, 2])[0])
        ```

    """
    gen = np.random.default_rng(seed=seed)
    return _rejection_sample(dims, True, max_draws, gen)

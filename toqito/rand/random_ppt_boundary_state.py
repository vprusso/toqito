"""Generates a random PPT density matrix close to the boundary of the PPT set."""

import numpy as np

from toqito.rand._ppt_sampling import _check_dims, _rejection_sample
from toqito.state_props.is_ppt import is_ppt


def random_ppt_boundary_state(
    dims: list[int], num_steps: int = 20, max_draws: int = 100_000, seed: int | None = None
) -> np.ndarray:
    r"""Generate a random PPT density matrix close to the boundary of the PPT set.

    A PPT state and an NPT state are drawn from the Hilbert-Schmidt measure by rejection, as in
    `toqito.rand.random_ppt_state` and `toqito.rand.random_npt_state`. The segment between them is then
    bisected `num_steps` times, each time replacing the endpoint that lies on the same side of the
    PPT boundary as the midpoint. The PPT endpoint is returned, so the result is PPT for every draw, and
    the smallest eigenvalue of its partial transpose is of order \(2^{-\text{num\_steps}}\): about
    \(10^{-7}\) after 20 steps.

    Every convex combination of two density matrices is a density matrix, so no renormalization is
    needed. As for `toqito.rand.random_ppt_state`, a PPT draw is not a bound entangled state, and the
    rejection step is only practical on small systems such as \(2 \otimes 2\) and \(2 \otimes 3\).

    Args:
        dims: The local dimensions \([d_A, d_B]\).
        num_steps: Number of bisection steps.
        max_draws: Maximum number of draws for each of the two rejection-sampled endpoints.
        seed: A seed used to instantiate numpy's random number generator.

    Returns:
        A \(d_A d_B\)-by-\(d_A d_B\) PPT density matrix near the PPT boundary.

    Raises:
        ValueError: If `dims` is not a list of two positive integers or `num_steps` is negative.
        RuntimeError: If no PPT or no NPT state is found within `max_draws` draws.

    Examples:
        The smallest eigenvalue of the partial transpose is positive but tiny.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import partial_transpose
        from toqito.rand import random_ppt_boundary_state
        rho = random_ppt_boundary_state([2, 2], num_steps=20, seed=42)
        print(0 <= np.linalg.eigvalsh(partial_transpose(rho, [1], [2, 2])).min() < 1e-5)
        ```

    """
    dim_a, dim_b = _check_dims(dims)
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    gen = np.random.default_rng(seed=seed)
    ppt_state = _rejection_sample(dims, True, max_draws, gen)
    npt_state = _rejection_sample(dims, False, max_draws, gen)
    for _ in range(num_steps):
        midpoint = (ppt_state + npt_state) / 2
        if is_ppt(midpoint, dim=[dim_a, dim_b]):
            ppt_state = midpoint
        else:
            npt_state = midpoint
    return ppt_state

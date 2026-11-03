"""Shared rejection sampler for random states selected by their partial transpose."""

import numpy as np

from toqito.rand.random_density_matrix import random_density_matrix
from toqito.state_props.is_ppt import is_ppt


def _check_dims(dims: list[int]) -> tuple[int, int]:
    """Validate a pair of local dimensions."""
    if len(dims) != 2 or any(int(d) != d or d < 1 for d in dims):
        raise ValueError("dims must be a list of two positive integers.")
    return int(dims[0]), int(dims[1])


def _rejection_sample(dims: list[int], want_ppt: bool, max_draws: int, gen: np.random.Generator) -> np.ndarray:
    """Draw Hilbert-Schmidt random density matrices until one has the requested partial-transpose class.

    Raises:
        RuntimeError: If no draw of the requested class is found within `max_draws` attempts.

    """
    dim_a, dim_b = _check_dims(dims)
    for _ in range(max_draws):
        rho = random_density_matrix(dim_a * dim_b, seed=int(gen.integers(0, 2**32 - 1)))
        if is_ppt(rho, dim=[dim_a, dim_b]) == want_ppt:
            return rho
    label = "PPT" if want_ppt else "NPT"
    raise RuntimeError(f"No {label} state found in {max_draws} draws; the {label} fraction is too small for rejection.")

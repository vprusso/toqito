"""Generates a Ginibre random matrix."""

import numpy as np


def random_ginibre(dim_n: int, dim_m: int, seed: int | None = None) -> np.ndarray:
    r"""Generate a Ginibre random matrix [@WikiCircLaw].

    Generates a random `dim_n`-by-`dim_m` Ginibre matrix.

    A *Ginibre random matrix* is a matrix with independent and identically distributed complex standard Gaussian
    entries.

    Ginibre random matrices are used in the construction of Wishart-random POVMs [@Heinosaari_2020_Random].

    Examples:

    Generate a random \(2\)-by-\(2\) Ginibre random matrix.

    ```python exec="1" source="above"
    from toqito.rand import random_ginibre
    
    print(random_ginibre(2, 2))
    ```

    It is also possible to pass a seed to this function for reproducibility.

    ```python exec="1" source="above"
    from toqito.rand import random_ginibre
    
    print(random_ginibre(2, 2, seed=42))
    ```

    Args:
        dim_n: The number of rows of the Ginibre random matrix.
        dim_m: The number of columns of the Ginibre random matrix.
        seed: A seed used to instantiate numpy's random number generator.

    Returns:
        A `dim_n`-by-`dim_m` Ginibre random matrix.

    """
    gen = np.random.default_rng(seed=seed)
    return (gen.standard_normal((dim_n, dim_m)) + 1j * gen.standard_normal((dim_n, dim_m))) / np.sqrt(2)

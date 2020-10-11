"""Generate Ginibre random matrix."""
import numpy as np


def random_ginibre(
    dim_n: int,
    dim_m: int,
) -> np.ndarray:
    r"""
    Generate a Ginibre random matrix [WIKCIRC]_.

    Generates a random :code:`dim_n`-by-:code:`dim_m` Ginibre matrix.

    A *Ginibre random matrix* is a matrix with independent and identically distributed complex
    standard Gaussian entries.

    Ginibre random matrices are used in the construction of Wishart-random POVMs [HMN20]_.

    Examples
    ==========

    Generate a random :math:`2`-by-:math:`2` Ginibre random matrix.

    >>> from toqito.random import random_ginibre
    >>> random_ginibre(2, 2)
    [[ 0.06037649-0.05158031j  0.46797859+0.21872729j]
     [-0.95223112-0.71959831j  0.3404352 +0.11166238j]]

    References
    ==========

    .. [WIKCIRC] Wikipedia: Circular law
        https://en.wikipedia.org/wiki/Circular_law

    .. [HMN20] Heinosaari, Teiko, Maria Anastasia Jivulescu, and Ion Nechita.
        "Random positive operator valued measures."
        Journal of Mathematical Physics 61.4 (2020): 042202.

    :param dim_n: The number of rows of the Ginibre random matrix.
    :param dim_m: The number of columns of the Ginibre random matrix.
    :return: A :code:`dim_n`-by-:code:`dim_m` Ginibre random density matrix.
    """
    return (np.random.randn(dim_n, dim_m) + 1j * np.random.randn(dim_n, dim_m)) / np.sqrt(2)

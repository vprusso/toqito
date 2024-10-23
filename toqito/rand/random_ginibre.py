"""Generates a Ginibre random matrix."""

import numpy as np


def random_ginibre(dim_n: int, dim_m: int, seed: int | None = None) -> np.ndarray:
    r"""Generate a Ginibre random matrix :cite:`WikiCircLaw`.

    Generates a random :code:`dim_n`-by-:code:`dim_m` Ginibre matrix.

    A *Ginibre random matrix* is a matrix with independent and identically distributed complex standard Gaussian
    entries.

    Ginibre random matrices are used in the construction of Wishart-random POVMs :cite:`Heinosaari_2020_Random`.

    Examples
    ==========

    Generate a random :math:`2`-by-:math:`2` Ginibre random matrix.

    >>> from toqito.rand import random_ginibre
    >>> random_ginibre(2, 2) # doctest: +SKIP
    array([[0.39166472-1.54657971j, 0.36538245+0.23324642j],
           [0.50103695-0.25857737j, 0.8357054 +0.31404353j]])

    It is also possible to pass a seed to this function for reproducibility.

    >>> from toqito.rand import random_ginibre
    >>> random_ginibre(2, 2, seed=42)
    array([[ 0.21546751-1.37959021j, -0.73537981-0.92077996j],
           [ 0.53064913+0.09039682j,  0.66507969-0.22361728j]])


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param dim_n: The number of rows of the Ginibre random matrix.
    :param dim_m: The number of columns of the Ginibre random matrix.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A :code:`dim_n`-by-:code:`dim_m` Ginibre random density matrix.

    """
    gen = np.random.default_rng(seed=seed)
    return (gen.standard_normal((dim_n, dim_m)) + 1j * gen.standard_normal((dim_n, dim_m))) / np.sqrt(2)

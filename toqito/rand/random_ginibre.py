"""Generate Ginibre random matrix."""

import numpy as np


def random_ginibre(dim_n: int, dim_m: int) -> np.ndarray:
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



    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param dim_n: The number of rows of the Ginibre random matrix.
    :param dim_m: The number of columns of the Ginibre random matrix.
    :return: A :code:`dim_n`-by-:code:`dim_m` Ginibre random density matrix.

    """
    return (np.random.randn(dim_n, dim_m) + 1j * np.random.randn(dim_n, dim_m)) / np.sqrt(2)

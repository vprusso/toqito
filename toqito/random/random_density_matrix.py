"""Generates a random density matrix."""
from typing import List, Union
import numpy as np
from toqito.random.random_unitary import random_unitary


def random_density_matrix(dim: int,
                          is_real: bool = False,
                          k_param: Union[List[int], int] = None,
                          distance_metric: str = "haar") -> np.ndarray:
    """
    Generates a random density matrix.

    Generates a random `dim`-by-`dim` density matrix distributed according to
    the Hilbert-Schmidt measure. The matrix is of rank <= `k_param` distributed
    according to the distribution `distance_metric` If `is_real = True`, then
    all of its entries will be real. The variable `distance_metric` must be one
    of:

        - `haar` (default):
            Generate a larger pure state according to the Haar measure and
            trace out the extra dimensions. Sometimes called the
            Hilbert-Schmidt measure when `k_param = dim`.

        - `bures`:
            The Bures measure.

    :param dim: The number of rows (and columns) of the density matrix.
    :param is_real: Boolean denoting whether the returned matrix will have all
                    real entries or not.
    :param k_param: Default value is equal to `dim`.
    :param distance_metric: The distance metric used to randomly generate the
                            density matrix. This metric is either the Haar
                            measure or the Bures measure. Default value is to
                            use the Haar measure.
    :return: A `dim`-by-`dim` random density matrix.
    """
    if k_param is None:
        k_param = dim

    # Haar / Hilbert-Schmidt measure.
    gin = np.random.rand(dim, k_param)

    if not is_real:
        gin = gin + 1j * np.random.rand(dim, k_param)

    if distance_metric == "bures":
        gin = np.matmul(random_unitary(dim, is_real) + np.identity(dim),  gin)

    rho = np.matmul(gin, np.matrix(gin).H)

    return np.divide(rho, np.trace(rho))


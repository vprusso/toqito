"""Generate random density matrix."""

import numpy as np

from toqito.rand import random_unitary


def random_density_matrix(
    dim: int,
    is_real: bool = False,
    k_param: list[int] | int = None,
    distance_metric: str = "haar",
) -> np.ndarray:
    r"""Generate a random density matrix.

    Generates a random :code:`dim`-by-:code:`dim` density matrix distributed according to the Hilbert-Schmidt measure.
    The matrix is of rank <= :code:`k_param` distributed according to the distribution :code:`distance_metric` If
    :code:`is_real = True`, then all of its entries will be real. The variable :code:`distance_metric` must be one of:

        - :code:`haar` (default):
            Generate a larger pure state according to the Haar measure and trace out the extra dimensions. Sometimes
            called the Hilbert-Schmidt measure when :code:`k_param = dim`.

        - :code:`bures`:
            The Bures measure.

    Examples
    ==========

    Using :code:`toqito`, we may generate a random complex-valued :math:`n`- dimensional density matrix. For
    :math:`d=2`, this can be accomplished as follows.

    >>> from toqito.rand import random_density_matrix
    >>> complex_dm = random_density_matrix(2)
    >>> complex_dm # doctest: +SKIP
    array([[0.83378447+0.j        , 0.25937396-0.12958422j],
           [0.25937396+0.12958422j, 0.16621553+0.j        ]])

    We can verify that this is in fact a valid density matrix using the :code:`is_denisty` function from :code:`toqito`
    as follows

    >>> from toqito.matrix_props import is_density
    >>> is_density(complex_dm)
    np.True_

    We can also generate random density matrices that are real-valued as follows.

    >>> from toqito.rand import random_density_matrix
    >>> real_dm = random_density_matrix(2, is_real=True)
    >>> real_dm # doctest: +SKIP
    array([[0.3686455 , 0.48243625],
           [0.48243625, 0.6313545 ]])


    Again, verifying that this is a valid density matrix can be done as follows.

    >>> from toqito.matrix_props import is_density
    >>> is_density(real_dm)
    np.True_

    By default, the random density operators are constructed using the Haar measure. We can select to generate the
    random density matrix according to the Bures metric instead as follows.

    >>> from toqito.rand import random_density_matrix
    >>> bures_mat = random_density_matrix(2, distance_metric="bures")
    >>> bures_mat # doctest: +SKIP
    array([[0.73969055+0.j        , 0.09494426-0.15279342j],
           [0.09494426+0.15279342j, 0.26030945+0.j        ]])


    As before, we can verify that this matrix generated is a valid density matrix.

    >>> from toqito.matrix_props import is_density
    >>> is_density(bures_mat)
    np.True_


    :param dim: The number of rows (and columns) of the density matrix.
    :param is_real: Boolean denoting whether the returned matrix will have all
                    real entries or not.
    :param k_param: Default value is equal to :code:`dim`.
    :param distance_metric: The distance metric used to randomly generate the
                            density matrix. This metric is either the Haar
                            measure or the Bures measure. Default value is to
                            use the Haar measure.
    :return: A :code:`dim`-by-:code:`dim` random density matrix.

    """
    if k_param is None:
        k_param = dim

    # Haar / Hilbert-Schmidt measure.
    gin = np.random.rand(dim, k_param)

    if not is_real:
        gin = gin + 1j * np.random.randn(dim, k_param)

    if distance_metric == "bures":
        gin = random_unitary(dim, is_real) + np.identity(dim) @ gin

    rho = gin @ np.array(gin).conj().T

    return np.divide(rho, np.trace(rho))

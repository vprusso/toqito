"""Generate random quantum states and measurements."""
from typing import List, Union

import numpy as np

from toqito.states import max_entangled
from toqito.perms import swap


__all__ = [
    "random_density_matrix",
    "random_ginibre",
    "random_povm",
    "random_state_vector",
    "random_unitary",
]


def random_density_matrix(
    dim: int,
    is_real: bool = False,
    k_param: Union[List[int], int] = None,
    distance_metric: str = "haar",
) -> np.ndarray:
    r"""
    Generate a random density matrix.

    Generates a random :code:`dim`-by-:code:`dim` density matrix distributed
    according to the Hilbert-Schmidt measure. The matrix is of rank <=
    :code:`k_param` distributed according to the distribution
    :code:`distance_metric` If :code:`is_real = True`, then all of its entries
    will be real. The variable :code:`distance_metric` must be one of:

        - :code:`haar` (default):
            Generate a larger pure state according to the Haar measure and
            trace out the extra dimensions. Sometimes called the
            Hilbert-Schmidt measure when :code:`k_param = dim`.

        - :code:`bures`:
            The Bures measure.

    Examples
    ==========

    Using :code:`toqito`, we may generate a random complex-valued :math:`n`-
    dimensional density matrix. For :math:`d=2`, this can be accomplished as
    follows.

    >>> from toqito.random import random_density_matrix
    >>> complex_dm = random_density_matrix(2)
    >>> complex_dm
    [[0.34903796+0.j       0.4324904 +0.103298j]
     [0.4324904 -0.103298j 0.65096204+0.j      ]]

    We can verify that this is in fact a valid density matrix using the
    :code:`is_denisty` function from :code:`toqito` as follows

    >>> from toqito.matrix_props import is_density
    >>> is_density(complex_dm)
    True

    We can also generate random density matrices that are real-valued as
    follows.

    >>> from toqito.random import random_density_matrix
    >>> real_dm = random_density_matrix(2, is_real=True)
    >>> real_dm
    [[0.37330805 0.46466224]
     [0.46466224 0.62669195]]

    Again, verifying that this is a valid density matrix can be done as follows.

    >>> from toqito.matrix_props import is_density
    >>> is_density(real_dm)
    True

    By default, the random density operators are constructed using the Haar
    measure. We can select to generate the random density matrix according to
    the Bures metric instead as follows.

    >>> from toqito.random import random_density_matrix
    >>> bures_mat = random_density_matrix(2, distance_metric="bures")
    >>> bures_mat
    [[0.59937164+0.j         0.45355087-0.18473365j]
     [0.45355087+0.18473365j 0.40062836+0.j        ]]

    As before, we can verify that this matrix generated is a valid density
    matrix.

    >>> from toqito.matrix_props import is_density
    >>> is_density(bures_mat)
    True

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
        gin = gin + 1j * np.random.rand(dim, k_param)

    if distance_metric == "bures":
        gin = np.matmul(random_unitary(dim, is_real) + np.identity(dim), gin)

    rho = np.matmul(gin, np.array(gin).conj().T)

    return np.divide(rho, np.trace(rho))


def random_ginibre(dim_n: int, dim_m: int,) -> np.ndarray:
    r"""
    Generate a Ginibre random matrix [WIKCIRC]_.

    Generates a random :code:`dim_n`-by-:code:`dim_m` Ginibre matrix.

    A *Ginibre random matrix* is a matrix with independent and identically
    distributed complex standard Gaussian entries.

    Ginibre random matrices are used in the construction of Wishart-random
    POVMs [HMN20]_.

    Examples
    ==========

    Generate a random 2-by-2 Ginibre random matrix.

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
    return (
        np.random.randn(dim_n, dim_m) + 1j * np.random.randn(dim_n, dim_m)
    ) / np.sqrt(2)


def random_povm(dim: int, num_inputs: int, num_outputs: int) -> np.ndarray:
    """
    Generate random positive operator valued measurements (POVMs) [WIKPOVM]_.

    Examples
    ==========

    We can generate a set of POVMs consisting of a specific dimension along with
    a given number of measurement inputs and measurement outputs. As an example,
    we can construct a random set of POVMs of dimension :math:`2` with :math:`2`
    inputs and :math:`2` outputs.

    >>> from toqito.random import random_povm
    >>> import numpy as np
    >>>
    >>> dim, num_inputs, num_outputs = 2, 2, 2
    >>> povms = random_povm(dim, num_inputs, num_outputs)
    >>> povms
    [[[[ 0.40313832+0.j,  0.59686168+0.j],
       [ 0.91134633+0.j,  0.08865367+0.j]],
     [[-0.27285707+0.j,  0.27285707+0.j],
      [-0.12086852+0.j,  0.12086852+0.j]]],
     [[[-0.27285707+0.j,  0.27285707+0.j],
      [-0.12086852+0.j,  0.12086852+0.j]],
     [[ 0.452533  +0.j,  0.547467  +0.j],
      [ 0.34692158+0.j,  0.65307842+0.j]]]]

    We can verify that this constitutes a valid set of POVM elements as checking
    that these operators all sum to the identity operator.

    >>> np.round(povms[:, :, 0, 0] + povms[:, :, 0, 1])
    [[1.+0.j, 0.+0.j],
     [0.+0.j, 1.+0.j]]

    References
    ==========
    .. [WIKPOVM] Wikipedia: POVM
        https://en.wikipedia.org/wiki/POVM

    :param dim: The dimension of the measurements.
    :param num_inputs: The number of inputs for the measurement.
    :param num_outputs: The number of outputs for the measurement.
    :return: A set of POVMs of dimension :code:`dim`.
    """
    povms = []
    gram_vectors = np.random.normal(size=(dim, dim, num_inputs, num_outputs))
    for input_block in gram_vectors:
        normalizer = sum(
            [
                np.matmul(np.array(output_block).T.conj(), output_block)
                for output_block in input_block
            ]
        )

        u_mat, d_mat, _ = np.linalg.svd(normalizer)

        output_povms = []
        for output_block in input_block:
            partial = (
                np.array(output_block, dtype=complex)
                .dot(u_mat)
                .dot(np.diag(d_mat ** (-1 / 2.0)))
            )
            internal = partial.dot(np.diag(np.ones(dim)) ** (1 / 2.0))
            output_povms.append(np.matmul(internal.T.conj(), internal))
        povms.append(output_povms)

    # This allows us to index the POVMs as [d, d, num_inputs, num_outputs].
    povms = np.swapaxes(np.array(povms), 0, 2)
    povms = np.swapaxes(povms, 1, 3)

    return povms


def random_state_vector(
    dim: Union[List[int], int], is_real: bool = False, k_param: int = 0
) -> np.ndarray:
    r"""Generate a random pure state vector.

    Examples
    ==========

    We may generate a random state vector. For instance, here is an example
    where we can generate a :math:`2`-dimensional random state vector.

    >>> from toqito.random import random_state_vector
    >>> vec = random_state_vector(2)
    >>> vec
    [[0.50993973+0.15292408j],
     [0.27787332+0.79960122j]]

    We can verify that this is in fact a valid state vector by computing the
    corresponding density matrix of the vector and checking if the density
    matrix is pure.

    >>> from toqito.state_props import is_pure
    >>> dm = vec.conj().T * vec
    >>> is_pure(dm)
    True

    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is :code:`False`.
    :param k_param: Default 0.
    :return: A :code:`dim`-by-:code:`dim` random unitary matrix.
    """
    # Schmidt rank plays a role.
    if 0 < k_param < np.min(dim):
        # Allow the user to enter a single number for dim.
        if isinstance(dim, int):
            dim = [dim, dim]

        # If you start with a separable state on a larger space and multiply
        # the extra `k_param` dimensions by a maximally entangled state, you
        # get a Schmidt rank `<= k_param` state.
        psi = max_entangled(k_param, True, False).toarray()

        a_param = np.random.rand(dim[0] * k_param, 1)
        b_param = np.random.rand(dim[1] * k_param, 1)

        if not is_real:
            a_param = a_param + 1j * np.random.rand(dim[0] * k_param, 1)
            b_param = b_param + 1j * np.random.rand(dim[1] * k_param, 1)

        mat_1 = np.kron(psi.conj().T, np.identity(int(np.prod(dim))))
        mat_2 = swap(
            np.kron(a_param, b_param),
            sys=[2, 3],
            dim=[k_param, dim[0], k_param, dim[1]],
        )

        ret_vec = mat_1 * mat_2
        return np.divide(ret_vec, np.linalg.norm(ret_vec))

    # Schmidt rank is full, so ignore it.
    ret_vec = np.random.rand(dim, 1)
    if not is_real:
        ret_vec = ret_vec + 1j * np.random.rand(dim, 1)
    return np.divide(ret_vec, np.linalg.norm(ret_vec))


def random_unitary(dim: Union[List[int], int], is_real: bool = False) -> np.ndarray:
    """
    Generate a random unitary or orthogonal matrix [MO09]_.

    Calculates a random unitary matrix (if :code:`is_real = False`) or a random
    real orthogonal matrix (if :code:`is_real = True`), uniformly distributed
    according to the Haar measure.

    Examples
    ==========

    We may generate a random unitary matrix. Here is an example of how we may
    be able to generate a random :math:`2`-dimensional random unitary matrix
    with complex entries.

    >>> from toqito.random import random_unitary
    >>> complex_dm = random_unitary(2)
    >>> complex_dm
    [[0.40563696+0.18092721j, 0.00066868+0.89594841j],
     [0.4237286 +0.78941628j, 0.27157521-0.35145826j]]

    We can verify that this is in fact a valid unitary matrix using the
    :code:`is_unitary` function from :code:`toqito` as follows

    >>> from toqito.matrix_props import is_unitary
    >>> is_unitary(complex_dm)
    True

    We can also generate random unitary matrices that are real-valued as
    follows.

    >>> from toqito.random import random_unitary
    >>> real_dm = random_unitary(2, True)
    >>> real_dm
    [[ 0.01972681, -0.99980541],
     [ 0.99980541,  0.01972681]]

    Again, verifying that this is a valid unitary matrix can be done as follows.

    >>> from toqito.matrix_props import is_unitary
    >>> is_unitary(real_dm)
    True

    We may also generate unitaries such that the dimension argument provided is
    a :code:`list` as opposed to an :code:`int`. Here is an example of a random
    unitary matrix of dimension :math:`4`.

    >>> from toqito.random import random_unitary
    >>> mat = random_unitary([4, 4], True)
    >>> mat
    [[ 0.48996358, -0.20978392,  0.56678587, -0.62823576],
     [ 0.62909119, -0.35852051, -0.68961425, -0.01181086],
     [ 0.38311399,  0.90865415, -0.1209574 , -0.11375677],
     [ 0.46626562, -0.04244265,  0.4342295 ,  0.76957113]]

    As before, we can verify that this matrix generated is a valid unitary
    matrix.

    >>> from toqito.matrix_props import is_unitary
    >>> is_unitary(mat)
    True

    References
    ==========
    .. [MO09] How to generate a random unitary matrix,
        Maris Ozols
        March 16, 2009,
        home.lu.lv/~sd20008/papers/essays/Random%20unitary%20%5Bpaper%5D.pdf

    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is :code:`False`.
    :return: A :code:`dim`-by-:code:`dim` random unitary matrix.
    """
    if isinstance(dim, int):
        dim = [dim, dim]

    # Construct the Ginibre ensemble.
    gin = np.random.rand(dim[0], dim[1])

    if not is_real:
        gin = gin + 1j * np.random.rand(dim[0], dim[1])

    # QR decomposition of the Ginibre ensemble.
    q_mat, r_mat = np.linalg.qr(gin)

    # Compute U from QR decomposition.
    r_mat = np.sign(np.diag(r_mat))

    # Protect against potentially zero diagonal entries.
    r_mat[r_mat == 0] = 1

    return np.matmul(q_mat, np.diag(r_mat))

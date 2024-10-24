"""Generates random quantum states using Qiskit."""


import numpy as np


def random_states(n: int, d: int, seed: int | None = None) -> list[np.ndarray]:
    r"""Generate a list of random quantum states.

    This function generates a list of quantum states, each of a specified dimension. The states are
    valid quantum states distributed according to the Haar measure.

    Examples
    ==========
    Generating three quantum states each of dimension 4.

    >>> from toqito.rand import random_states
    >>> states = random_states(3, 4)
    >>> len(states)
    3
    >>> states[0].shape
    (4, 1)
    >>> states  # doctest: +SKIP
    [array([[-0.2150583 +3.12920500e-01j],
           [-0.45427289-2.42799455e-01j],
           [ 0.34457387-3.20987030e-05j],
           [ 0.47739088-4.93844159e-01j]]), array([[ 0.05596192-0.40459234j],
           [-0.8497132 +0.06357884j],
           [-0.22808131-0.16261183j],
           [-0.16047978+0.05386145j]]), array([[ 0.12592373+0.00508266j],
           [-0.71527467+0.41425908j],
           [-0.27852449+0.39980357j],
           [ 0.17033502+0.18562365j]])]

    It is also possible to pass a seed to this function for reproducibility.

    >>> from toqito.rand import random_states
    >>> states = random_states(3, 4, seed=42)
    >>> states
    [array([[ 0.13830446+0.0299699j ],
           [-0.47202619+0.51163029j],
           [ 0.34061349+0.21219233j],
           [ 0.42690188-0.39001418j]]), array([[-0.71489214+0.1351165j ],
           [-0.47714049-0.35135073j],
           [ 0.04684288+0.32187898j],
           [-0.11587661-0.01829369j]]), array([[-0.00827473-0.0910465j ],
           [-0.42013238-0.33536439j],
           [ 0.43311201+0.60211343j],
           [ 0.38307005-0.07610726j]])]



    :param n: int
        The number of random states to generate.
    :param d: int
        The dimension of each quantum state.
    :param seed: int | None
        A seed used to instantiate numpy's random number generator.

    :return: list[numpy.ndarray]
        A list of `n` numpy arrays, each representing a d-dimensional quantum state as a
        column vector.

    """
    gen = np.random.default_rng(seed=seed)
    samples = gen.normal(size=(n, d)) + 1j * gen.normal(size=(n, d))
    samples /= np.linalg.norm(samples, axis=1)[:, np.newaxis]
    return [sample.reshape(-1, 1) for sample in samples]

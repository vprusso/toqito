"""Generates random quantum states using Qiskit."""

import numpy as np


def random_states(n: int, d: int, seed: int | None = None) -> list[np.ndarray]:
    r"""Generate a list of random quantum states.

    This function generates a list of quantum states, each of a specified dimension. The states are
    valid quantum states distributed according to the Haar measure.

    Examples
    ==========
    Generating three quantum states each of dimension 4.

    .. jupyter-execute::

     from toqito.rand import random_states

     states = random_states(3, 4)
     print(f"length of states is {len(states)}")

     print(f"Shape of each state vector: {states[0].shape}")

     for idx, state in enumerate(states):
        print(f"\nState {idx}:")
        print(state)

    It is also possible to pass a seed to this function for reproducibility.

    .. jupyter-execute::

     from toqito.rand import random_states

     states = random_states(3, 4, seed=42)

     for idx, state in enumerate(states):
        print(f"\nState {idx}:")
        print(state)



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

"""Generate random POVM."""
import numpy as np


def random_povm(dim: int, num_inputs: int, num_outputs: int) -> np.ndarray:
    """
    Generate random positive operator valued measurements (POVMs) [WIKPOVM]_.

    Examples
    ==========

    We can generate a set of POVMs consisting of a specific dimension along with a given number of
    measurement inputs and measurement outputs. As an example, we can construct a random set of
    POVMs of dimension :math:`2` with :math:`2` inputs and :math:`2` outputs.

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

    We can verify that this constitutes a valid set of POVM elements as checking that these
    operators all sum to the identity operator.

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
            [np.array(output_block).T.conj() @ output_block for output_block in input_block]
        )

        u_mat, d_mat, _ = np.linalg.svd(normalizer)

        output_povms = []
        for output_block in input_block:
            partial = (
                np.array(output_block, dtype=complex).dot(u_mat).dot(np.diag(d_mat ** (-1 / 2.0)))
            )
            internal = partial.dot(np.diag(np.ones(dim)) ** (1 / 2.0))
            output_povms.append(internal.T.conj() @ internal)
        povms.append(output_povms)

    # This allows us to index the POVMs as [d, d, num_inputs, num_outputs].
    povms = np.swapaxes(np.array(povms), 0, 2)
    povms = np.swapaxes(povms, 1, 3)

    return povms

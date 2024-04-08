"""Generate random POVM."""

import numpy as np


def random_povm(dim: int, num_inputs: int, num_outputs: int) -> np.ndarray:
    """Generate random positive operator valued measurements (POVMs) :cite:`WikiPOVM`.

    Examples
    ==========

    We can generate a set of `dim`-by-`dim` POVMs consisting of a specific dimension along with a given number of
    measurement inputs and measurement outputs. As an example, we can construct a random set of :math:`2`-by-:math:`2`
    POVMs of dimension with :math:`2` inputs and :math:`2` outputs.

    >>> from toqito.rand import random_povm
    >>> import numpy as np
    >>>
    >>> dim, num_inputs, num_outputs = 2, 2, 2
    >>> povms = random_povm(dim, num_inputs, num_outputs)
    >>> povms  # doctest: +SKIP
    array([[[[ 0.20649603+0.j,  0.79350397+0.j],
             [ 0.77451456+0.j,  0.22548544+0.j]],
    <BLANKLINE>
            [[-0.25971638+0.j,  0.25971638+0.j],
             [-0.28048509+0.j,  0.28048509+0.j]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[-0.25971638+0.j,  0.25971638+0.j],
             [-0.28048509+0.j,  0.28048509+0.j]],
    <BLANKLINE>
            [[ 0.40448792+0.j,  0.59551208+0.j],
             [ 0.10740892+0.j,  0.89259108+0.j]]]])


    We can verify that this constitutes a valid set of POVM elements as checking that these operators all sum to the
    identity operator.

    >>> np.round(povms[:, :, 0, 0] + povms[:, :, 0, 1]) # doctest: +SKIP
    [[1.+0.j, 0.+0.j],
     [0.+0.j, 1.+0.j]]


    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :param dim: The dimensions of the measurements.
    :param num_inputs: The number of inputs for the measurement.
    :param num_outputs: The number of outputs for the measurement.
    :return: A set of `dim`-by-`dim` POVMs of shape `(dim, dim, num_inputs, num_outputs)`.

    """
    povms = []
    gram_vectors = np.random.normal(size=(num_inputs, num_outputs, dim, dim))
    for input_block in gram_vectors:
        normalizer = sum(np.array(output_block).T.conj() @ output_block for output_block in input_block)
        u_mat, d_mat, _ = np.linalg.svd(normalizer)

        output_povms = []
        for output_block in input_block:
            partial = np.array(output_block, dtype=complex).dot(u_mat).dot(np.diag(d_mat ** (-1 / 2.0)))
            internal = partial.dot(np.diag(np.ones(dim)) ** (1 / 2.0))
            output_povms.append(internal.T.conj() @ internal)
        povms.append(output_povms)

    # This allows us to index the POVMs as [dim, dim, num_inputs, num_outputs].
    povms = np.swapaxes(np.array(povms), 0, 2)
    povms = np.swapaxes(povms, 1, 3)

    return povms

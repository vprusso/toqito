"""Generates a random POVM."""

import numpy as np


def random_povm(dim: int, num_inputs: int, num_outputs: int, seed: int | None = None) -> np.ndarray:
    """Generate random positive operator valued measurements (POVMs) :footcite:`WikiPOVM`.

    Examples
    ==========

    We can generate a set of `dim`-by-`dim` POVMs consisting of a specific dimension along with a given number of
    measurement inputs and measurement outputs. As an example, we can construct a random set of :math:`2`-by-:math:`2`
    POVMs of dimension with :math:`2` inputs and :math:`2` outputs.

    .. jupyter-execute::

     import numpy as np
     from toqito.rand import random_povm

     dim, num_inputs, num_outputs = 2, 2, 2

     povms = random_povm(dim, num_inputs, num_outputs)

     povms


    We can verify that this constitutes a valid set of POVM elements as checking that these operators all sum to the
    identity operator.

    .. jupyter-execute::

     np.round(povms[:, :, 0, 0] + povms[:, :, 0, 1])

    It is also possible to add a seed for reproducibility.

    .. jupyter-execute::

     import numpy as np
     from toqito.rand import random_povm

     dim, num_inputs, num_outputs = 2, 2, 2

     povms = random_povm(dim, num_inputs, num_outputs, seed=42)

     povms

    We can once again verify that this constitutes a valid set of POVM elements as checking that
    these operators all sum to the identity operator.

    .. jupyter-execute:

     np.round(povms[:, :, 0, 0] + povms[:, :, 0, 1])

    References
    ==========
    .. footbibliography::




    :param dim: The dimensions of the measurements.
    :param num_inputs: The number of inputs for the measurement.
    :param num_outputs: The number of outputs for the measurement.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A set of `dim`-by-`dim` POVMs of shape `(dim, dim, num_inputs, num_outputs)`.

    """
    povms = []
    gen = np.random.default_rng(seed=seed)
    gram_vectors = gen.normal(size=(num_inputs, num_outputs, dim, dim))
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

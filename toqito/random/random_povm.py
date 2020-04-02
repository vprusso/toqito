"""Generate a random set of positive-operator-valued measurements (POVMs)."""
import numpy as np


def random_povm(dim: int, num_inputs: int, num_outputs: int) -> np.ndarray:
    """
    Generate random positive operator valued measurements (POVMs).

    References:
        [1] Wikipedia: POVM
        https://en.wikipedia.org/wiki/POVM

    :param dim: The dimension of the measurements.
    :param num_inputs: The number of inputs for the measurement.
    :param num_outputs: The number of outputs for the measurement.
    :return: A set of POVMs of dimension `dim`.
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

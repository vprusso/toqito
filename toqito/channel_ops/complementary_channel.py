"""Computes the complementary channel/map of a superoperator."""

import numpy as np


def complementary_channel(kraus_ops: list[np.ndarray]) -> list[np.ndarray]:
    r"""Compute the Kraus operators for the complementary map of a quantum channel.

    (Section: Representations and Characterizations of Channels from :cite:`Watrous_2018_TQI`).

    The complementary map is derived from the given quantum channel's Kraus operators by
    rearranging the rows of the input Kraus operators into the Kraus operators of the
    complementary map.

    Specifically, for each Kraus operator :math:`K_i` in the input channel :math:`\Phi`,
    we define the complementary Kraus operators :math:`K_i^C` by stacking the rows of
    :math:`K_i` from all Kraus operators vertically.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If the input is not a valid list of Kraus operators.
    :param kraus_ops: A list of numpy arrays representing the Kraus operators of a quantum channel.
                      Each Kraus operator is assumed to be a square matrix.
    :return: A list of numpy arrays representing the Kraus operators of the complementary map.

    """
    num_kraus = len(kraus_ops)
    op_dim = kraus_ops[0].shape[0]

    if any(k.shape[0] != k.shape[1] for k in kraus_ops):
        raise ValueError("All Kraus operators must be square matrices.")

    comp_kraus_ops = []

    for row in range(op_dim):
        comp_kraus_op = np.vstack([kraus_ops[i][row, :] for i in range(num_kraus)])
        comp_kraus_ops.append(comp_kraus_op)

    return comp_kraus_ops

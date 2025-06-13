"""Computes the maximum probability of distinguishing two quantum channels."""

import numpy as np

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import kraus_to_choi
from toqito.helper import channel_dim


def channel_distinguishability(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    p: list[float],
    dim: int | list[int] | np.ndarray = None,
) -> float:
    r"""Compute the optimal probability of distinguishing two quantum channels.

    Bayesian discrimination of two quantum channels is implemented where a priori
    probabilities are provided. (Section 3.3.3 of :cite:`Watrous_2018_TQI`).
    Implementation in QETLAB :cite: `QETLAB_link` is used.


    Examples
    ========
    To compute the optimal probability of distinguishing two quantum channels,

    .. jupyter-execute::

    from toqito.channels import amplitude_damping
    from toqito.channel_ops import kraus_to_choi
    from toqito.channel_metrics import channel_distinguishability
    # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
    choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25))
    choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5))

    p = [0.5, 0.5]

    dist = channel_distinguishability(choi_ch_1, choi_ch_1, p)


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If channels have different input or output dimensions.
    :raises ValueError: If prior probabilities do not add up to 1.
    :raises ValueError: If number of prior probabilities not equal to 2.
    :param phi: A superoperator. It should be provided either as a Choi matrix,
                or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
    :param psi: A superoperator. It should be provided either as a Choi matrix,
                or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
    :param p: A list. Prior probabilities of the two channels.
    :param dim: Input and output dimensions of the channels.
    :return: The optimal probability of discriminating two quantum channels.

    """
    # Get the input, output and environment dimensions of phi and psi.
    d_in_phi, d_out_phi, d_e = channel_dim(phi, dim)
    d_in_psi, d_out_psi, d_e = channel_dim(psi, dim)

    # If the variable `phi` and/or `psi` are provided as a list, we assume this is a list
    # of Kraus operators. We convert to choi matrices if not provided as choi matrix.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    if isinstance(psi, list):
        psi = kraus_to_choi(psi)

    dim_phi, dim_psi = (d_in_phi, d_out_phi), (d_in_psi, d_out_psi)

    # checking for errors.
    if dim_phi != dim_psi:
        raise ValueError("The channels must have the same dimension input and output spaces as each other.")

    if len(p) != 2:
        raise ValueError("p must be a probability distribution with 2 entries.")

    if max(p) >= 1:
        return 1

    if (abs(sum(p) - 1)) != 0:
        raise ValueError("Sum of prior probabilities must add up to 1.")

    prob = 1 / 2 * (1 + completely_bounded_trace_norm(p[0] * phi - p[1] * psi))
    return prob

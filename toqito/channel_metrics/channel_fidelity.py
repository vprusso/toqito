"""Compute the channel fidelity between two quantum channels."""
import cvxpy
import numpy as np

from toqito.channels import partial_trace


def channel_fidelity(choi_1: np.ndarray, choi_2: np.ndarray) -> float:
    r"""
    Compute the channel fidelity between two quantum channels [VW20]_.

    Let :math:`\Phi : \text{L}(\mathcal{Y}) \rightarrow \text{L}(\mathcal{X})` and
    :math:`\Psi: \text{L}(\mathcal{Y}) \rightarrow \text{L}(\mathcal{X})` be quantum channels. Then
    the root channel fidelity defined as

    .. math::
        \sqrt{F}(\Phi, \Psi) := \text{inf}_{\rho} \sqrt{F}(\Phi(\rho), \Psi(\rho))

    where :math:`\rho \in \text{D}(\mathcal{Z} \otimes \mathcal{X})` can be calculated by means of
    the following semidefinite program (Proposition 50) in [VW20]_,

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \lambda \\
            \text{subject to:} \quad & \lambda \mathbb{I}_{\mathcal{Z}} \leq
                \text{Re}\left( \text{Tr}_{\mathcal{Y}} \left( Q \right) \right),\\
                & \begin{pmatrix}
                    J(\Phi) & Q^* \\
                    Q & J(\Psi)
                \end{pmatrix} \geq 0
        \end{align*}

    where :math:`Q \in \text{L}(\mathcal{Z} \otimes \mathcal{X})`.

    Examples
    ========

    For two identical channels, we should expect that the channel fidelity should yield a value of
    :math:`1`.

    >>> from toqito.channels import dephasing
    >>> from toqito.channel_metrics import channel_fidelity
    >>>
    >>> # The Choi matrices of dimension-4 for the dephasing channel
    >>> choi_1 = dephasing(4)
    >>> choi_2 = dephasing(4)
    >>> channel_fidelity(choi_1, choi_2)
    0.9999790499568767

    We can also compute the channel fidelity between two different channels. For example, we can
    compute the channel fidelity between the dephasing and depolarizing channels.

    >>> from toqito.channels import dephasing, depolarizing
    >>> from toqito.channel_metrics import channel_fidelity
    >>>
    >>> # The Choi matrices of dimension-4 for the dephasing and depolarizing channels
    >>> choi_1 = dephasing(4)
    >>> choi_2 = depolarizing(4)
    >>> channel_fidelity(choi_1, choi_2)
    0.5001368672503087

    References
    ==========
    .. [VW20] Katariya, Vishal, and Mark M. Wilde.
        "Geometric distinguishability measures limit quantum channel estimation and discrimination."
        https://arxiv.org/abs/2004.10708

    :param choi_1: The Choi matrix of the first quantum channel.
    :param choi_2: The Choi matrix of the second quantum channel.
    :return: The channel fidelity between the channels specified by the quantum channels
             corresponding to the Choi matrices :code:`choi_1` and :code:`choi_2`.
    """
    if choi_1.shape != choi_2.shape:
        raise ValueError("The Choi matrices provided should be of equal dimension.")

    choi_dim_x, choi_dim_y = choi_1.shape
    if choi_dim_x != choi_dim_y:
        raise ValueError("The Choi matrix provided must be square.")

    choi_dim = choi_dim_x
    dim = int(np.log2(choi_dim))

    lam = cvxpy.Variable(nonneg=True)
    q_var = cvxpy.Variable((choi_dim, choi_dim), complex=True)

    constraints = []
    objective = cvxpy.Maximize(lam)

    constraints.append(cvxpy.bmat([[choi_1, q_var.H], [q_var, choi_2]]) >> 0)

    constraints.append(lam * np.identity(dim) <= cvxpy.real(partial_trace(q_var, [2], [dim, dim])))

    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()

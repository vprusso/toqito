"""Computes the maximum probability of distinguishing two quantum channels."""

import numpy as np
import picos as pc

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import kraus_to_choi
from toqito.helper import channel_dim


def channel_distinguishability(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    p: list[float] | None,
    dim: int | list[int] | np.ndarray = None,
    strategy: str = "bayesian",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs,
) -> float:
    r"""Compute the optimal probability of distinguishing two quantum channels.

    Bayesian and minimax discrimination of two quantum channels are implemented.

    For Bayesian discrimination, channels to be distinguished should have a given a priori probability distribution.
    The task of discriminating channels can be connected to the completely bounded trace norm
    (Section 3.3.3 of :footcite:`Watrous_2018_TQI`).
    The problem is finding POVMs for which error probability of discrimination of
    output states is minimized after input state is acted on by the two quantum channels.
    In the language of statistical decision theory, the problem is equivalent to minimizing quantum Bayes' risk.

    In the minimax problem, there are no a priori probabilities.
    Minimax discrimination of two channels consists of finding the
    optimal input state so that the two possible output states are discriminated
    with minimum risk. (:footcite:`d2005minimax`).

    QETLAB's functionality inspired the Bayesian option  :footcite:`QETLAB_link`
    and the minimax option is adapted from  QuTIpy :footcite:`QuTIpy_link`.


    Examples
    ========
    Optimal probability of distinguishing two amplitude damping channels in the Bayesian setting:

    .. jupyter-execute::

        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_metrics import channel_distinguishability
        # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5))

        p = [0.5, 0.5]

        channel_distinguishability(choi_ch_1, choi_ch_2, p)

    Optimal probability of distinguishing two amplitude damping channels in the minimax setting:

    .. jupyter-execute::

        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_metrics import channel_distinguishability
        # Define two amplitude damping channels with gamma=0.25 and gamma=0.5
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5))

        channel_distinguishability(choi_ch_1, choi_ch_2, None, [2, 2], strategy="minimax",
                        primal_dual="primal")


    References
    ==========
    .. footbibliography::

    :raises ValueError: If prior probabilities not provided at all for Bayesian strategy.
    :raises ValueError: If strategy is neither Bayesian nor minimax.
    :raises ValueError: If channels have different input or output dimensions.
    :raises ValueError: If prior probabilities do not add up to 1.
    :raises ValueError: If number of prior probabilities not equal to 2.
    :param phi: A superoperator. It should be provided either as a Choi matrix,
                or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
    :param psi: A superoperator. It should be provided either as a Choi matrix,
                or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
    :param p: Prior probabilities of the two channels.
    :param dim: Input and output dimensions of the channels.
    :param strategy: Whether to perform Bayesian or minimax discrimination task. Possible
                     values are "Bayesian" and "minimax". Defualt option is :code:`strategy="Bayesian"`.
    :param solver: Optimization option for :code:`picos` solver. Default option is :code:`solver="cvxopt"`.
    :param primal_dual: Option for the optimization problem. Defualt option is :code:`solver="cvxopt"`.
    :param kwargs: Additional arguments to pass to picos' solve method.
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

    dim_phi, dim_psi = np.array([d_in_phi, d_out_phi]), np.array([d_in_psi, d_out_psi])

    # checking for errors.
    if strategy.lower() not in ("bayesian", "minimax"):
        raise ValueError("The strategy must either be Bayesian or Minimax.")

    if not np.array_equal(dim_phi, dim_psi):
        raise ValueError("The channels must have the same dimension input and output spaces as each other.")

    if strategy.lower() == "bayesian":
        if p is None:
            raise ValueError("Must provide valid prior probabilities for Bayesian strategy.")

        if len(p) != 2:
            raise ValueError("p must be a probability distribution with 2 entries.")

        if max(p) >= 1:
            return 1

        if abs(sum(p) - 1) != 0:
            raise ValueError("Sum of prior probabilities must add up to 1.")

        # optimal success probability is minimizing error probability (Bayes risk).
        return 1 / 2 * (1 + completely_bounded_trace_norm(p[0] * phi - p[1] * psi))

    if primal_dual == "primal":
        return _minimax_primal(phi, psi, d_in_phi[0], d_out_phi[0], solver=solver, **kwargs)

    return _minimax_dual(phi, psi, d_in_phi[0], d_out_phi[0], solver=solver, **kwargs)


def _minimax_dual(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    dimA: int,
    dimB: int,
    solver: str = "cvxopt",
    **kwargs,
) -> float:
    """Find the dual problem for minimax quantum channel distinguishability SDP."""
    J_var = list([phi, psi])

    problem = pc.Problem()

    a_var = pc.RealVariable("a", lower=0)
    P_var = pc.RealVariable("P", 2)
    Y_var = pc.HermitianVariable("Y", (dimA * dimB, dimA * dimB))

    Y0 = pc.partial_trace(Y_var, 1)

    problem.add_list_of_constraints(Y_var >> P_var[i] * J_var[i] for i in range(2))
    problem.add_constraint(pc.sum(P_var) == 1)
    problem.add_constraint(Y_var >> 0)
    problem.add_constraint(Y0 == a_var * np.eye(dimA))
    problem.add_list_of_constraints(p >= 0 for p in P_var)

    problem.set_objective("min", a_var)

    problem.solve(solver=solver, **kwargs)

    return problem.value


def _minimax_primal(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    psi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    dimA: int,
    dimB: int,
    solver: str = "cvxopt",
    **kwargs,
) -> float:
    """Find the primal problem for minimax quantum channel distinguishability SDP."""
    J_var = list([phi, psi])

    problem = pc.Problem()

    a_var = pc.RealVariable("a", lower=0)
    P_var = [pc.HermitianVariable(f"P[{i}]", (dimA * dimB, dimA * dimB)) for i in range(2)]
    rho = pc.HermitianVariable("rho", (dimA, dimA))

    problem.add_list_of_constraints(P_var[i] >> 0 for i in range(2))
    problem.add_list_of_constraints(a_var <= pc.trace(P_var[i] * J_var[i]).real for i in range(2))
    problem.add_constraint(pc.sum(P_var) == rho @ np.eye(dimB))
    problem.add_constraint(rho >> 0)
    problem.add_constraint(pc.trace(rho) == 1)

    problem.set_objective("max", a_var)

    problem.solve(solver=solver, **kwargs)

    return problem.value

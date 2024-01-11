"""State distinguishability."""
import numpy as np
import picos

from toqito.state_ops import pure_to_mixed


def state_distinguishability(
    vectors: list[np.ndarray],
    probs: list[float] = None,
    solver: str = "cvxopt",
    primal_dual: str = "dual",
) -> float:
    r"""Compute probability of state distinguishability :cite:`Eldar_2003_SDPApproach`.

    The "quantum state distinguishability" problem involves a collection of :math:`n` quantum states

    .. math::
        \rho = \{ \rho_0, \ldots, \rho_n \},

    as well as a list of corresponding probabilities

    .. math::
        p = \{ p_0, \ldots, p_n \}.

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i` Bob
    wants to guess which state he was given from the collection of states.

    One can specify the distinguishability method using the :code:`dist_method` argument.

    For :code:`dist_method = "min-error"`, this is the default method that yields the probability of
    distinguishing quantum states that minimize the probability of error.

    For :code:`dist_method = "unambiguous"`, Alice and Bob never provide an incorrect answer,
    although it is possible that their answer is inconclusive.

    When :code:`dist_method = "min-error"`, this function implements the following semidefinite
    program that provides the optimal probability with which Bob can conduct quantum state
    distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
            \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                     & M_0, \ldots, M_n \geq 0
        \end{align*}

    When :code:`dist_method = "unambiguous"`, this function implements the following semidefinite
    program that provides the optimal probability with which Bob can conduct unambiguous quantum
    state distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
            \text{subject to:} \quad & M_0 + \ldots + M_{n+1} = \mathbb{I},\\
                                     & \langle M_i, \rho_j \rangle = 0,
                                       \quad 1 \leq i, j \leq n, \quad i \not= j, \\
                                     & M_0, \ldots, M_n \geq 0.
        \end{align*}

    Examples
    ==========

    State distinguishability for two state density matrices.
    In this example, the states :math:`|0\rangle` and :math:`|1\rangle`
    are orthogonal and therefore perfectly distinguishable.

    >>> from toqito.states import basis
    >>> from toqito.state_opt import state_distinguishability
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00 = e_0 * e_0.conj().T
    >>> e_11 = e_1 * e_1.conj().T
    >>> states = [e_00, e_11]
    >>> probs = [1 / 2, 1 / 2]
    >>> res, _ = state_distinguishability(states, probs)
    1.000

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param vectors: A list of states provided as vectors.
    :param probs: Respective list of probabilities each state is selected. If no
                  probabilities are provided, a uniform probability distribution is assumed.
    :param solver: Optimization option for `picos` solver. Default option is `solver_option="cvxopt"`.
    :param primal_dual: Option for the optimization problem.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states` along with the optimal set of measurements.
    """
    if not all(vector.shape == vectors[0].shape for vector in vectors):
        raise ValueError("Vectors for state distinguishability must all have the same dimension.")

    # Assumes a uniform probabilities distribution among the states if one is not explicitly provided.
    n = vectors[0].shape[0]
    probs = [1 / n] * n if probs is None else probs

    if primal_dual == "primal":
        return _min_error_primal(vectors=vectors, probs=probs, solver=solver)
    return _min_error_dual(vectors=vectors, probs=probs, solver=solver)


def _min_error_primal(
        vectors: list[np.ndarray],
        probs: list[float] = None,
        solver: str = "cvxopt",
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the primal problem for minimum-error quantum state distinguishability SDP."""
    n, dim = len(vectors), vectors[0].shape[0]

    problem = picos.Problem()
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(n)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(picos.sum(measurements) == picos.I(dim))

    problem.set_objective(
        "max",
        picos.sum(
            [
                (probs[i] * pure_to_mixed(vectors[i].reshape(-1, 1)) | measurements[i])
                for i in range(n)
            ]
        ),
    )
    solution = problem.solve(solver=solver)
    return solution.value, measurements


def _min_error_dual(
        vectors: list[np.ndarray],
        probs: list[float] = None,
        solver: str = "cvxopt"
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the dual problem for minimum-error quantum state distinguishability SDP."""
    n, dim = len(vectors), vectors[0].shape[0]
    problem = picos.Problem()

    # Set up variables and constraints for SDP:
    y_var = picos.HermitianVariable("Y", (dim, dim))
    problem.add_list_of_constraints(
        [
            y_var >> probs[i] * pure_to_mixed(vector.reshape(-1, 1))
            for i, vector in enumerate(vectors)
        ]
    )

    # Objective function:
    problem.set_objective("min", picos.trace(y_var))
    solution = problem.solve(solver=solver, primals=None)

    measurements = [problem.get_constraint(k).dual for k in range(n)]

    return solution.value, measurements

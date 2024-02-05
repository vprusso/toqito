"""State distinguishability."""
import numpy as np
import picos

from toqito.matrix_ops import calculate_vector_matrix_dimension
from toqito.matrix_props import has_same_dimension
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

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i`. Bob
    wants to guess which state he was given from the collection of states.

    This function implements the following semidefinite program that provides the optimal probability with which Bob can
    conduct quantum state distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
            \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                     & M_0, \ldots, M_n \geq 0
        \end{align*}

    Examples
    ==========

    State distinguishability for the Bell states (which are perfectly distinguishable).

    >>> from toqito.states import bell
    >>> from toqito.state_opt import state_distinguishability
    >>> states = [bell(0), bell(1), bell(2), bell(3)]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>> res, _ = state_distinguishability(vectors=states, probs=probs, primal_dual="dual")
    0.9999999994695794

    Note that if we are just interested in obtaining the optimal value, it is computationally less intensive to compute
    the dual problem over the primal problem. However, the primal problem does allow us to extract the explicit
    measurement operators which may be of interest to us.

    >>> from toqito.states import bell
    >>> from toqito.state_opt import state_distinguishability
    >>> states = [bell(0), bell(1), bell(2), bell(3)]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>> res, measurements = state_distinguishability(vectors=states, probs=probs, primal_dual="primal")
    >>> np.around(measurements[0], decimals=5)
    [[ 0.5+0.j -0. +0.j  0. +0.j  0.5-0.j]
     [-0. -0.j  0. +0.j -0. -0.j -0. -0.j]
     [ 0. -0.j -0. +0.j  0. +0.j  0. -0.j]
     [ 0.5+0.j -0. +0.j  0. +0.j  0.5+0.j]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param vectors: A list of states provided as vectors.
    :param probs: Respective list of probabilities each state is selected. If no
                  probabilities are provided, a uniform probability distribution is assumed.
    :param solver: Optimization option for `picos` solver. Default option is `solver_option="cvxopt"`.
    :param primal_dual: Option for the optimization problem. Default option is `"dual"`.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states` along with the optimal set of measurements.

    """
    if not has_same_dimension(vectors):
        raise ValueError("Vectors for state distinguishability must all have the same dimension.")

    # Assumes a uniform probabilities distribution among the states if one is not explicitly provided.
    n = len(vectors)
    probs = [1 / n] * n if probs is None else probs
    dim = calculate_vector_matrix_dimension(vectors[0])

    if primal_dual == "primal":
        return _min_error_primal(vectors=vectors, dim=dim, probs=probs, solver=solver)
    return _min_error_dual(vectors=vectors, dim=dim, probs=probs, solver=solver)


def _min_error_primal(
        vectors: list[np.ndarray],
        dim: int,
        probs: list[float] = None,
        solver: str = "cvxopt",
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the primal problem for minimum-error quantum state distinguishability SDP."""
    n = len(vectors)

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
        dim: int,
        probs: list[float] = None,
        solver: str = "cvxopt"
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the dual problem for minimum-error quantum state distinguishability SDP."""
    n = len(vectors)
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

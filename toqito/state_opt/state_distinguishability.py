"""State distinguishability."""

import numpy as np
import picos

from toqito.matrix_ops import calculate_vector_matrix_dimension, vector_to_density_matrix
from toqito.matrix_props import has_same_dimension


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
    >>> '%.2f' % res
    '1.00'

    .. note::
        You do not need to use `'%.2f' %` when you use this function.

        We use this to format our output such that `doctest` compares the calculated output to the
        expected output upto two decimal points only. The accuracy of the solvers can calculate the
        `float` output to a certain amount of precision such that the value deviates after a few digits
        of accuracy.

    Note that if we are just interested in obtaining the optimal value, it is computationally less intensive to compute
    the dual problem over the primal problem. However, the primal problem does allow us to extract the explicit
    measurement operators which may be of interest to us.

    >>> import numpy as np
    >>> from toqito.states import bell
    >>> from toqito.state_opt import state_distinguishability
    >>> states = [bell(0), bell(1), bell(2), bell(3)]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>> res, measurements = state_distinguishability(vectors=states, probs=probs, primal_dual="primal")
    >>> np.around(measurements[0], decimals=5)  # doctest: +SKIP
    array([[ 0.5+0.j,  0. +0.j, -0. -0.j,  0.5-0.j],
           [ 0. -0.j,  0. +0.j, -0. +0.j,  0. -0.j],
           [-0. +0.j, -0. -0.j,  0. +0.j, -0. +0.j],
           [ 0.5+0.j,  0. +0.j, -0. -0.j,  0.5+0.j]])

    .. note::
        We use `#doctest: +SKIP` here to stop `doctest` from comparing the expected output
        to the calculated output. Depending on the accuracy of the solvers, the expected value
        might not be exactly as what's shown here.

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
    num_measurements = len(vectors)
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(num_measurements)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(picos.sum(measurements) == picos.I(dim))

    dms = [vector_to_density_matrix(vector) for vector in vectors]

    problem.set_objective("max", np.real(picos.sum([(probs[i] * dms[i] | measurements[i]) for i in range(n)])))
    solution = problem.solve(solver=solver)
    return solution.value, measurements


def _min_error_dual(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] = None,
    solver: str = "cvxopt",
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the dual problem for minimum-error quantum state distinguishability SDP."""
    n = len(vectors)
    problem = picos.Problem()

    # Set up variables and constraints for SDP:
    y_var = picos.HermitianVariable("Y", (dim, dim))
    problem.add_list_of_constraints(
        [y_var >> probs[i] * vector_to_density_matrix(vector) for i, vector in enumerate(vectors)]
    )

    # Objective function:
    problem.set_objective("min", picos.trace(y_var))
    solution = problem.solve(solver=solver, primals=None)

    measurements = [problem.get_constraint(k).dual for k in range(n)]

    return solution.value, measurements

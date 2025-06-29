"""Calculates the probability of optimally distinguishing quantum states."""

import numpy as np
import picos

from toqito.matrix_ops import calculate_vector_matrix_dimension, to_density_matrix, vectors_to_gram_matrix
from toqito.matrix_props import has_same_dimension


def state_distinguishability(
    vectors: list[np.ndarray],
    probs: list[float] = None,
    strategy: str = "min_error",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable] | tuple[picos.HermitianVariable] | tuple[picos.RealVariable]]:
    r"""Compute probability of state distinguishability :footcite:`Eldar_2003_SDPApproach`.

    The "quantum state distinguishability" problem involves a collection of :math:`n` quantum states

    .. math::
        \rho = \{ \rho_1, \ldots, \rho_n \},

    as well as a list of corresponding probabilities

    .. math::
        p = \{ p_1, \ldots, p_n \}.

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i`. Bob
    wants to guess which state he was given from the collection of states.

    For :code:`strategy = "min_error"`, this is the default method that yields the minimal
    probability of error for Bob.

    In that case, this function implements the following semidefinite program that provides the
    optimal probability with which Bob can conduct quantum state distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
            \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                     & M_0, \ldots, M_n \geq 0.
        \end{align*}

    For :code:`strategy = "unambiguous"`, Bob never provides an incorrect answer, although it is
    possible that his answer is inconclusive.

    In that case, this function implements the following semidefinite program that provides the
    optimal probability with which Bob can conduct unambiguous quantum state distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \mathbf{p} \cdot \mathbf{q} \\
            \text{subject to:} \quad & \Gamma - Q \geq 0,\\
                                     & \mathbf{q} \geq 0
        \end{align*}

    .. math::
        \begin{align*}
            \text{minimize:} \quad & \text{Tr}(\Gamma Z) \\
            \text{subject to:} \quad & z_i + p_i + \text{Tr}\left(F_iZ\right)=0,\\
                                     & Z, z \geq 0
        \end{align*}

    where :math:`\mathbf{p}` is the vector whose :math:`i`-th coordinate contains the probability
    that the state is prepared in state :math:`\left|\psi_i\right\rangle`, :math:`\Gamma` is
    the Gram matrix of :math:`\left|\psi_1\right\rangle,\cdots,\left|\psi_n\right\rangle` and :math:`F_i` is
    :math:`-|i\rangle\langle i|`.

    .. warning::
        Note that it only makes sense to distinguish unambiguously when the pure states are linearly
        independent. Calling this function on a set of states that doesn't verify this property will
        return 0.

    Examples
    ==========

    Minimal-error state distinguishability for the Bell states (which are perfectly distinguishable).

    .. jupyter-execute::

     import numpy as np
     from toqito.states import bell
     from toqito.state_opt import state_distinguishability

     states = [bell(0), bell(1), bell(2), bell(3)]
     probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

     res, _ = state_distinguishability(vectors=states, probs=probs, primal_dual="dual")

     np.around(res, decimals=2)

    Note that if we are just interested in obtaining the optimal value, it is computationally less intensive to compute
    the dual problem over the primal problem. However, the primal problem does allow us to extract the explicit
    measurement operators which may be of interest to us.

    .. jupyter-execute::

     import numpy as np
     from toqito.states import bell
     from toqito.state_opt import state_distinguishability

     states = [bell(0), bell(1), bell(2), bell(3)]
     probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

     res, measurements = state_distinguishability(vectors=states, probs=probs, primal_dual="primal")

     np.around(measurements[0], decimals=5)

    Unambiguous state distinguishability for unbiased states.

    .. jupyter-execute::

     import numpy as np
     from toqito.state_opt import state_distinguishability

     states = [np.array([[1.], [0.]]), np.array([[1.],[1.]]) / np.sqrt(2)]
     probs = [1 / 2, 1 / 2]

     res, _ = state_distinguishability(vectors=states, probs=probs, primal_dual="primal", strategy="unambiguous")

     np.around(res, decimals=2)

    References
    ==========
    .. footbibliography::



    :param vectors: A list of states provided as vectors.
    :param probs: Respective list of probabilities each state is selected. If no
                  probabilities are provided, a uniform probability distribution is assumed.
    :param strategy: Whether to perform unambiguous or minimal error discrimination task. Possible
                     values are "min_error" and "unambiguous". Default option is `strategy="min_error"`.
    :param solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
    :param primal_dual: Option for the optimization problem. Default option is `"dual"`.
    :param kwargs: Additional arguments to pass to picos' solve method.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states` along with the optimal set of measurements.

    """
    if not has_same_dimension(vectors):
        raise ValueError("Vectors for state distinguishability must all have the same dimension.")

    # Assumes a uniform probabilities distribution among the states if one is not explicitly provided.
    n = len(vectors)
    probs = [1 / n] * n if probs is None else probs
    dim = calculate_vector_matrix_dimension(vectors[0])

    if strategy == "min_error":
        if primal_dual == "primal":
            return _min_error_primal(vectors=vectors, dim=dim, probs=probs, solver=solver, **kwargs)
        return _min_error_dual(vectors=vectors, dim=dim, probs=probs, solver=solver, **kwargs)

    if primal_dual == "primal":
        return _unambiguous_primal(vectors=vectors, probs=probs, solver=solver, **kwargs)

    return _unambiguous_dual(vectors=vectors, probs=probs, solver=solver, **kwargs)


def _min_error_primal(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the primal problem for minimum-error quantum state distinguishability SDP."""
    n = len(vectors)

    problem = picos.Problem()
    num_measurements = len(vectors)
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(num_measurements)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(picos.sum(measurements) == picos.I(dim))

    dms = [to_density_matrix(vector) for vector in vectors]

    problem.set_objective("max", np.real(picos.sum([(probs[i] * dms[i] | measurements[i]) for i in range(n)])))
    solution = problem.solve(solver=solver, **kwargs)
    return solution.value, measurements


def _min_error_dual(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the dual problem for minimum-error quantum state distinguishability SDP."""
    n = len(vectors)
    problem = picos.Problem()

    # Set up variables and constraints for SDP:
    y_var = picos.HermitianVariable("Y", (dim, dim))
    problem.add_list_of_constraints([y_var >> probs[i] * to_density_matrix(vector) for i, vector in enumerate(vectors)])

    # Objective function:
    problem.set_objective("min", picos.trace(y_var))
    solution = problem.solve(solver=solver, primals=None, **kwargs)

    measurements = [problem.get_constraint(k).dual for k in range(n)]

    return solution.value, measurements


def _unambiguous_primal(
    vectors: list[np.ndarray],
    probs: list[float] = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, tuple[picos.RealVariable]]:
    """Solve the primal problem for unambiguous quantum state distinguishability SDP.

    Implemented according to Equation (5) of :footcite:`Gupta_2024_Unambiguous`:.
    """
    n = len(vectors)
    problem = picos.Problem()

    gram = vectors_to_gram_matrix(vectors)
    success_probabilities = picos.RealVariable("success_probabilities", n, lower=0)

    problem.add_constraint(gram - picos.diag(success_probabilities) >> 0)
    problem.set_objective("max", np.array(probs) | success_probabilities)

    problem.solve(solver=solver, **kwargs)

    return problem.value, (success_probabilities,)


def _unambiguous_dual(
    vectors: list[np.ndarray],
    probs: list[float] = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, tuple[picos.HermitianVariable]]:
    """Solve the dual problem for unambiguous quantum state distinguishability SDP.

    Implemented according to Equation (5) of :footcite:`Gupta_2024_Unambiguous`.
    """
    n = len(vectors)
    problem = picos.Problem()

    gram = vectors_to_gram_matrix(vectors)
    lagrangian_variable_big_z = picos.SymmetricVariable("Z", (n, n))

    problem.add_constraint(lagrangian_variable_big_z >> 0)
    problem.add_list_of_constraints(lagrangian_variable_big_z[i, i] >= probs[i] for i in range(n))

    problem.set_objective("min", picos.trace(gram * lagrangian_variable_big_z))

    problem.solve(solver=solver, **kwargs)

    return problem.value, (lagrangian_variable_big_z,)

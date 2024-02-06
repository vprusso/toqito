"""State exclusion."""
import numpy as np
import picos

from toqito.matrix_ops import calculate_vector_matrix_dimension
from toqito.matrix_props import has_same_dimension
from toqito.state_ops import pure_to_mixed


def state_exclusion(
    vectors: list[np.ndarray],
    probs: list[float] = None,
    solver: str = "cvxopt",
    primal_dual: str = "dual",
) -> tuple[float, list[picos.HermitianVariable]]:
    r"""Compute probability of single state conclusive state exclusion.

    The *quantum state exclusion* problem involves a collection of :math:`n` quantum states

    .. math::
        \rho = \{ \rho_0, \ldots, \rho_n \},

    as well as a list of corresponding probabilities

    .. math::
        p = \{ p_0, \ldots, p_n \}.

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i`.

    Bob wants to guess which state he was *not* given from the collection of states. State exclusion implies that
    ability to discard (with certainty) at least one out of the "n" possible quantum states by applying a measurement.

    This function implements the following semidefinite program that provides the optimal probability with which Bob can
    conduct quantum state exclusion.

        .. math::
            \begin{equation}
                \begin{aligned}
                    \text{minimize:} \quad & \sum_{i=1}^n p_i \langle M_i, \rho_i \rangle \\
                    \text{subject to:} \quad & \sum_{i=1}^n M_i = \mathbb{I}_{\mathcal{X}}, \\
                                             & M_0, \ldots, M_n \in \text{Pos}(\mathcal{X}).
                \end{aligned}
            \end{equation}

        .. math::
            \begin{equation}
                \begin{aligned}
                    \text{maximize:} \quad & \text{Tr}(Y)
                    \text{subject to:} \quad & Y \leq M_1, \\
                                             & Y \leq M_2, \\
                                             & \vdots \\
                                             & Y \leq M_n, \\
                                             & Y \text{Herm}(\mathcal{X}).
                \end{aligned}
            \end{equation}


    The conclusive state exclusion SDP is written explicitly in :cite:`Bandyopadhyay_2014_Conclusive`. The problem
    of conclusive state exclusion was also thought about under a different guise in :cite:`Pusey_2012_On`.

    Examples
    ==========

    Consider the following two Bell states

    .. math::
        \begin{equation}
            \begin{aligned}
                u_0 &= \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), \\
                u_1 &= \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).
            \end{aligned}
        \end{equation}

    It is not possible to conclusively exclude either of the two states. We can see that the result of the function in
    :code:`toqito` yields a value of :math:`0` as the probability for this to occur.

    >>> from toqito.state_opt import state_exclusion
    >>> from toqito.states import bell
    >>> import numpy as np
    >>>
    >>> vectors = [bell(0), bell(1)]
    >>> probs = [1/2, 1/2]
    >>>
    >>> state_exclusion(vectors, probs)
    1.6824720366950206e-09

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
    """Find the primal problem for minimum-error quantum state exclusion SDP."""
    n = len(vectors)

    problem = picos.Problem()
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(n)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(picos.sum(measurements) == picos.I(dim))

    problem.set_objective(
        "min",
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
    """Find the dual problem for minimum-error quantum state exclusion SDP."""
    n = len(vectors)
    problem = picos.Problem()

    # Set up variables and constraints for SDP:
    y_var = picos.HermitianVariable("Y", (dim, dim))
    problem.add_list_of_constraints(
        [
            y_var << probs[i] * pure_to_mixed(vector.reshape(-1, 1))
            for i, vector in enumerate(vectors)
        ]
    )

    # Objective function:
    problem.set_objective("max", picos.trace(y_var))
    solution = problem.solve(solver=solver)

    measurements = [problem.get_constraint(k).dual for k in range(n)]

    return solution.value, measurements

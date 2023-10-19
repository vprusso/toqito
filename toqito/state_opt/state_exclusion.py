"""State exclusion."""
import numpy as np
import picos

from toqito.state_ops import pure_to_mixed


def state_exclusion(
    vectors: list[np.ndarray], probs: list[float] = None, solver: str = "cvxopt", primal_dual="dual"
) -> tuple[float, list[picos.HermitianVariable]]:
    r"""
    Compute probability of single state exclusion.

    The *quantum state exclusion* problem involves a collection of :math:`n`
    quantum states

    .. math::
        \rho = \{ \rho_0, \ldots, \rho_n \},

    as well as a list of corresponding probabilities

    .. math::
        p = \{ p_0, \ldots, p_n \}.

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state
    :math:`\rho_i`.

    Bob wants to guess which state he was *not* given from the collection of
    states. State exclusion implies that ability to discard (with certainty) at
    least one out of the "n" possible quantum states by applying a measurement.

    This function implements the following semidefinite program that provides
    the optimal probability with which Bob can conduct quantum state exclusion.

        .. math::
            \begin{equation}
                \begin{aligned}
                    \text{minimize:} \quad & \sum_{i=0}^n p_i \langle M_i,
                                                \rho_i \rangle \\
                    \text{subject to:} \quad & M_0 + \ldots + M_n =
                                               \mathbb{I}, \\
                                             & M_0, \ldots, M_n >= 0.
                \end{aligned}
            \end{equation}

    The conclusive state exclusion SDP is written explicitly in [BJOP14]_. The problem of conclusive
    state exclusion was also thought about under a different guise in [PBR12]_.

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

    It is not possible to conclusively exclude either of the two states. We can see that the result
    of the function in :code:`toqito` yields a value of :math:`0` as the probability for this to
    occur.

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
    .. [PBR12] "On the reality of the quantum state"
        Pusey, Matthew F., Barrett, Jonathan, and Rudolph, Terry.
        Nature Physics 8.6 (2012): 475-478.
        arXiv:1111.3328

    .. [BJOP14] "Conclusive exclusion of quantum states"
        Bandyopadhyay, Somshubhro, Jain, Rahul, Oppenheim, Jonathan,
        Perry, Christopher
        Physical Review A 89.2 (2014): 022336.
        arXiv:1306.4683

    :param states: A list of states provided as vectors.
    :param probs: Respective list of probabilities each state is selected. If no
                  probabilities are provided, a uniform probability distribution is assumed.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states` along with the optimal set of measurements.
    """
    if primal_dual == "primal":
        return _min_error_primal(vectors, probs, solver)
    else:
        return _min_error_dual(vectors, probs, solver)


def _min_error_primal(vectors: list[np.ndarray], probs: list[float] = None, solver: str = "cvxopt"):
    """Find the primal problem for minimum-error quantum state exclusion SDP."""
    n, dim = len(vectors), vectors[0].shape[0]
    if probs is None:
        probs = [1 / len(vectors)] * len(vectors)

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
    vectors: list[np.ndarray], probs: list[float] = None, solver: str = "cvxopt"
) -> float:
    """Find the dual problem for minimum-error quantum state exclusion SDP."""
    dim = vectors[0].shape[0]
    if probs is None:
        probs = [1 / len(vectors)] * len(vectors)

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

    measurements = [problem.get_constraint(k).dual for k in range(len(vectors))]

    return solution.value, measurements

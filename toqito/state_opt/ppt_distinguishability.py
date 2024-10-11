"""PPT state distinguishability."""

import numpy as np
import picos

from toqito.matrix_ops import calculate_vector_matrix_dimension, to_density_matrix
from toqito.matrix_props import has_same_dimension


def ppt_distinguishability(
    vectors: list[np.ndarray],
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float] = None,
    strategy: str = "min_error",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
) -> float:
    r"""Compute probability of optimally distinguishing a state via PPT measurements :cite:`Cosentino_2013_PPT`.

    Implements the semidefinite program (SDP) whose optimal value is equal to the maximum
    probability of perfectly distinguishing orthogonal maximally entangled states using any PPT
    measurement; a measurement whose operators are positive under partial transpose. This SDP was
    explicitly provided in :cite:`Cosentino_2013_PPT`.

    One can specify the distinguishability method using the :code:`dist_method` argument.

    For :code:`dist_method = "min_error"`, this is the default method that yields the probability of
    distinguishing quantum states via PPT measurements that minimize the probability of error.

    For :code:`dist_method = "unambig"`, Alice and Bob never provide an incorrect answer,
    although it is possible that their answer is inconclusive.

    For more info, go to the tutorial in the documentation :ref:`ref-label-state-dist-ppt`.

    Examples
    ==========

    Consider the following Bell states:

    .. math::
        \begin{equation}
            \begin{aligned}
            |\psi_0 \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, &\quad
            |\psi_1 \rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \\
            |\psi_2 \rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}, &\quad
            |\psi_3 \rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}.
            \end{aligned}
        \end{equation}

    It was illustrated in :cite:`Yu_2012_Four` that for the following set of states

    .. math::
        \begin{equation}
            \begin{aligned}
            \rho_1^{(2)} &= |\psi_0 \rangle | \psi_0 \rangle \langle \psi_0 | \langle \psi_0 |, \quad
            \rho_2^{(2)} &= |\psi_1 \rangle | \psi_3 \rangle \langle \psi_1 | \langle \psi_3 |, \\
            \rho_3^{(2)} &= |\psi_2 \rangle | \psi_3 \rangle \langle \psi_2 | \langle \psi_3 |, \quad
            \rho_4^{(2)} &= |\psi_3 \rangle | \psi_3 \rangle \langle \psi_3 | \langle \psi_3 |, \\
            \end{aligned}
        \end{equation}

    that the optimal probability of distinguishing via a PPT measurement should yield
    :math:`7/8 \approx 0.875` as was proved in :cite:`Yu_2012_Four`.

    >>> import numpy as np
    >>> from toqito.states import bell
    >>> from toqito.state_opt import ppt_distinguishability
    >>> # Bell vectors:
    >>> psi_0 = bell(0)
    >>> psi_1 = bell(2)
    >>> psi_2 = bell(3)
    >>> psi_3 = bell(1)
    >>>
    >>> # YDY vectors from :cite:`Yu_2012_Four`.
    >>> x_1 = np.kron(psi_0, psi_0)
    >>> x_2 = np.kron(psi_1, psi_3)
    >>> x_3 = np.kron(psi_2, psi_3)
    >>> x_4 = np.kron(psi_3, psi_3)
    >>>
    >>> # YDY density matrices.
    >>> rho_1 = x_1 @ x_1.conj().T
    >>> rho_2 = x_2 @ x_2.conj().T
    >>> rho_3 = x_3 @ x_3.conj().T
    >>> rho_4 = x_4 @ x_4.conj().T
    >>>
    >>> states = [rho_1, rho_2, rho_3, rho_4]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>>
    >>> opt_val, _ = ppt_distinguishability(vectors=states, probs=probs, dimensions=[2, 2, 2, 2], subsystems=[0, 2])
    >>> '%.3f' % opt_val
    '0.875'

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param vectors: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param subsystems: A list of integers that correspond to the complex Euclidean space dimensions.
    :param dimensions: A list of integers that correspond to the dimensions of the subsystems.
    :param strategy: The method of distinguishing states.
    :param solver: The SDP solver to use.
    :param primal_dual: Option for the optimization problem.
    :return: The optimal probability with which the states can be distinguished
             via PPT measurements.

    """
    if not has_same_dimension(vectors):
        raise ValueError("Vectors for state distinguishability must all have the same dimension.")

    # Assumes a uniform probabilities distribution among the states if one is not explicitly provided.
    n = len(vectors)
    probs = [1 / n] * n if probs is None else probs

    if primal_dual == "primal":
        return _min_error_primal(
            vectors=vectors, subsystems=subsystems, dimensions=dimensions, probs=probs, solver=solver, strategy=strategy
        )
    return _min_error_dual(
        vectors=vectors, subsystems=subsystems, dimensions=dimensions, probs=probs, solver=solver, strategy=strategy
    )


def _min_error_primal(
    vectors: list[np.ndarray],
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float],
    solver: str = "cvxopt",
    strategy: str = "min_error",
):
    """Primal problem for the SDP with PPT constraints."""
    n = len(vectors)
    d = calculate_vector_matrix_dimension(vectors[0])

    problem = picos.Problem()
    num_measurements = n if strategy == "min_error" else n + 1
    measurements = [picos.HermitianVariable(f"M[{i}]", (d, d)) for i in range(num_measurements)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(picos.sum(measurements) == picos.I(d))

    # Add PPT constraint.
    problem.add_list_of_constraints(
        [
            picos.partial_transpose(
                meas,
                subsystems=subsystems,
                dimensions=dimensions,
            )
            >> 0
            for meas in measurements
        ]
    )

    dms = [to_density_matrix(vector) for vector in vectors]
    if strategy == "unambig":
        for i in range(n):
            for j in range(n):
                if i != j:
                    problem.add_constraint(probs[j] * dms[j] | measurements[i] == 0)

    problem.set_objective("max", np.real(picos.sum([probs[i] * (dms[i] | measurements[i]) for i in range(n)])))
    solution = problem.solve(solver=solver)

    return solution.value, measurements


def _min_error_dual(
    vectors: list[np.ndarray],
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float],
    solver: str = "cvxopt",
    strategy: str = "min_error",
):
    """Semidefinite program with PPT constraints (dual problem)."""
    d = vectors[0].shape[0]

    if strategy != "min_error":
        raise ValueError("Minimum-error PPT distinguishability only supported at this time.")

    problem = picos.Problem()
    q_vars = [picos.HermitianVariable(f"Q[{i}]", (d, d)) for i in range(len(vectors))]

    y_var = picos.HermitianVariable("Y", (d, d))
    problem.add_list_of_constraints(
        [
            y_var - probs[i] * to_density_matrix(vectors[i])
            >> picos.partial_transpose(
                q_var,
                subsystems=subsystems,
                dimensions=dimensions,
            )
            for i, q_var in enumerate(q_vars)
        ]
    )
    problem.add_list_of_constraints([q_var >> 0 for q_var in q_vars])

    problem.set_objective("min", picos.trace(y_var))
    solution = problem.solve(solver=solver)

    measurements = [problem.get_constraint(k).dual for k in range(len(vectors))]
    return solution.value, measurements

"""Calculates the probability of error of single state conclusive state exclusion."""

from typing import Any

import numpy as np
import picos

from toqito.matrix_ops import calculate_vector_matrix_dimension, to_density_matrix
from toqito.matrix_props import has_same_dimension


def state_exclusion(
    vectors: list[np.ndarray],
    probs: list[float] | None = None,
    strategy: str = "min_error",
    measurement: str = "positive",
    subsystems: list[int] | None = None,
    dimensions: list[int] | None = None,
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs: Any,
) -> tuple[float, list[picos.HermitianVariable] | tuple[picos.HermitianVariable, picos.RealVariable]]:
    r"""Compute probability of error of single state conclusive state exclusion.

    The *quantum state exclusion* problem involves a collection of \(n\) quantum states

    \[
        \rho = \{ \rho_0, \ldots, \rho_n \},
    \]

    as well as a list of corresponding probabilities

    \[
        p = \{ p_0, \ldots, p_n \}.
    \]

    Alice chooses \(i\) with probability \(p_i\) and creates the state \(\rho_i\).

    Bob wants to guess which state he was *not* given from the collection of states. State exclusion implies that
    ability to discard at least one out of the "n" possible quantum states by applying a measurement.

    For `strategy = "min_error"`, this is the default method that yields the minimal probability of error for Bob.

    In that case, this function implements the following semidefinite program that provides the optimal probability
    with which Bob can conduct quantum state exclusion.

    \[
        \begin{equation}
            \begin{aligned}
                \text{minimize:} \quad & \sum_{i=1}^n p_i \langle M_i, \rho_i \rangle \\
                \text{subject to:} \quad & \sum_{i=1}^n M_i = \mathbb{I}_{\mathcal{X}}, \\
                                            & M_0, \ldots, M_n \in \text{Pos}(\mathcal{X}).
            \end{aligned}
        \end{equation}
    \]

    \[
        \begin{equation}
            \begin{aligned}
                \text{maximize:} \quad & \text{Tr}(Y) \\
                \text{subject to:} \quad & Y \preceq p_1\rho_1, \\
                                            & Y \preceq p_2\rho_2, \\
                                            & \vdots \\
                                            & Y \preceq p_n\rho_n, \\
                                            & Y \in\text{Herm}(\mathcal{X}).
            \end{aligned}
        \end{equation}
    \]

    If `measurement = "ppt"`, the measurement operators are also constrained to be positive under partial transpose on
    the requested subsystem(s). In the primal minimum-error problem, this adds the constraints

    \[
        \Gamma_j(M_i) \succeq 0
    \]

    for every measurement operator \(M_i\) and for the subsystem(s) \(j\) listed in `subsystems`, where \(\Gamma_j\)
    is the partial transpose map with subsystem dimensions `dimensions`. The corresponding dual uses the dual PPT cone,
    \(\text{Pos} + \Gamma_j(\text{Pos})\), so the constraints become

    \[
        p_i\rho_i - Y \in \text{Pos} + \Gamma_j(\text{Pos}).
    \]

    For `strategy = "unambiguous"`, Bob never provides an incorrect answer, although it is
    possible that his answer is inconclusive. This function then yields the probability of an inconclusive outcome.

    In that case, this function implements the following semidefinite program that provides the
    optimal probability with which Bob can conduct unambiguous quantum state distinguishability.

    \[
        \begin{align*}
            \text{minimize:} \quad & \text{Tr}\left(
                \left(\sum_{i=1}^n p_i\rho_i\right)\left(\mathbb{I}-\sum_{i=1}^nM_i\right)
                \right) \\
            \text{subject to:} \quad & \sum_{i=1}^nM_i \preceq \mathbb{I},\\
                                     & M_1, \ldots, M_n \succeq 0, \\
                                     & \langle M_1, \rho_1 \rangle, \ldots, \langle M_n, \rho_n \rangle =0
        \end{align*}
    \]

    \[
        \begin{align*}
            \text{maximize:} \quad & 1 - \text{Tr}(N) \\
            \text{subject to:} \quad & a_1p_1\rho_1, \ldots, a_np_n\rho_n \succeq \sum_{i=1}^np_i\rho_i - N,\\
                                     & N \succeq 0,\\
                                     & a_1, \ldots, a_n \in\mathbb{R}
        \end{align*}
    \]


    !!! Note
        This function supports both pure states (vectors) and mixed states (density matrices).
        It is known that it is always possible to perfectly exclude pure states that are linearly dependent.
        Thus, calling this function on a set of states with this property will return 0.

    The conclusive state exclusion SDP is written explicitly in [@bandyopadhyay2014conclusive]. The problem
    of conclusive state exclusion was also thought about under a different guise in [@pusey2012reality].

    Args:
        vectors: A list of states provided as vectors (for pure states) or density matrices (for mixed states).
        probs: Respective list of probabilities each state is selected. If no probabilities are provided, a uniform
            probability distribution is assumed.
        strategy: Whether to perform minimal error or unambiguous discrimination task. Possible values are "min_error"
            and "unambiguous". Both strategies support pure and mixed states.
        measurement: The type of measurement to use. Possible values are "positive" (default) for standard positive
            measurements and "ppt" for PPT (positive partial transpose) measurements.
        subsystems: A list of integers specifying which subsystems to transpose for PPT measurements. Required when
            `measurement="ppt"`.
        dimensions: A list of integers specifying the dimensions of each subsystem. Required when `measurement="ppt"`.
        solver: Optimization option for `picos` solver. Default option is `solver_option="cvxopt"`.
        primal_dual: Option for the optimization problem.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        The optimal probability with which Bob can guess the state he was not given from `states` along with the optimal
        set of measurements.

    Examples:
        Consider the following two Bell states

        \[
            \begin{equation}
                \begin{aligned}
                    u_0 &= \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), \\
                    u_1 &= \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).
                \end{aligned}
            \end{equation}
        \]

        It is not possible to conclusively exclude either of the two states. We can see that the result of the
        function in
        `|toqito⟩` yields a value of \(0\) as the probability for this to occur.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import bell
        from toqito.state_opt import state_exclusion

        vectors = [bell(0), bell(1)]
        probs = [1/2, 1/2]

        print(np.around(state_exclusion(vectors, probs)[0], decimals=2))
        ```

        Unambiguous state exclusion for unbiased pure states.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt import state_exclusion

        states = [np.array([[1.], [0.]]), np.array([[1.],[1.]]) / np.sqrt(2)]

        res, _ = state_exclusion(states, primal_dual="primal", strategy="unambiguous", abs_ipm_opt_tol=1e-5)

        print(np.around(res, decimals=2))
        ```

        State exclusion for mixed states.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_opt import state_exclusion

        # Two mixed states
        rho1 = 0.7 * np.array([[1., 0.], [0., 0.]]) + 0.3 * np.eye(2) / 2
        rho2 = 0.7 * np.array([[0., 0.], [0., 1.]]) + 0.3 * np.eye(2) / 2
        states = [rho1, rho2]

        res, _ = state_exclusion(states, primal_dual="dual")

        print(np.around(res, decimals=2))
        ```

        PPT-constrained minimum-error state exclusion.

        ```python
        import numpy as np
        from toqito.state_opt import state_exclusion
        from toqito.states import bell

        states = [bell(0), bell(1), bell(2)]

        res, _ = state_exclusion(
            states,
            measurement="ppt",
            subsystems=[0],
            dimensions=[2, 2],
            primal_dual="primal",
            cvxopt_kktsolver="ldl",
        )

        print(np.around(res, decimals=2))
        ```

        !!! Note
            If you encounter a `ZeroDivisionError` or an `ArithmeticError` when using cvxopt as a solver (which is the
            default), you might want to set the `abs_ipm_opt_tol` option to a lower value (the default being `1e-8`) or
            to set the `cvxopt_kktsolver` option to `ldl`.

            See https://gitlab.com/picos-api/picos/-/issues/341

    """
    if not has_same_dimension(vectors):
        raise ValueError("Vectors for state distinguishability must all have the same dimension.")

    # Assumes a uniform probabilities distribution among the states if one is not explicitly provided.
    n = len(vectors)
    probs = [1 / n] * n if probs is None else probs
    dim = calculate_vector_matrix_dimension(vectors[0])

    if measurement == "ppt":
        _validate_ppt_params(dim=dim, subsystems=subsystems, dimensions=dimensions)
        if strategy == "min_error":
            if primal_dual == "primal":
                return _ppt_min_error_primal(
                    vectors=vectors,
                    dim=dim,
                    subsystems=subsystems,
                    dimensions=dimensions,
                    probs=probs,
                    solver=solver,
                    **kwargs,
                )
            return _ppt_min_error_dual(
                vectors=vectors,
                dim=dim,
                subsystems=subsystems,
                dimensions=dimensions,
                probs=probs,
                solver=solver,
                **kwargs,
            )
        if primal_dual == "primal":
            return _ppt_unambiguous_primal(
                vectors=vectors,
                dim=dim,
                subsystems=subsystems,
                dimensions=dimensions,
                probs=probs,
                solver=solver,
                **kwargs,
            )
        return _ppt_unambiguous_dual(
            vectors=vectors,
            dim=dim,
            subsystems=subsystems,
            dimensions=dimensions,
            probs=probs,
            solver=solver,
            **kwargs,
        )

    if measurement != "positive":
        raise ValueError("Argument `measurement` must be either 'positive' or 'ppt'.")

    if strategy == "min_error":
        if primal_dual == "primal":
            return _min_error_primal(vectors=vectors, dim=dim, probs=probs, solver=solver, **kwargs)
        return _min_error_dual(vectors=vectors, dim=dim, probs=probs, solver=solver, **kwargs)

    if primal_dual == "primal":
        return _unambiguous_primal(vectors=vectors, dim=dim, probs=probs, solver=solver, **kwargs)

    return _unambiguous_dual(vectors=vectors, dim=dim, probs=probs, solver=solver, **kwargs)


def _validate_ppt_params(dim: int, subsystems: list[int] | None, dimensions: list[int] | None) -> None:
    """Validate subsystem information for PPT-constrained measurements."""
    if subsystems is None or dimensions is None:
        raise ValueError("The 'subsystems' and 'dimensions' parameters are required for PPT measurements.")

    if np.prod(dimensions) != dim:
        raise ValueError("The product of `dimensions` must equal the dimension of the states.")

    if any(sys < 0 or sys >= len(dimensions) for sys in subsystems):
        raise ValueError("Entries of `subsystems` must index into `dimensions`.")


def _ppt_dual_cone_constraint(
    problem: picos.Problem,
    expr: picos.expressions.ComplexAffineExpression,
    name: str,
    subsystems: list[int],
    dimensions: list[int],
) -> None:
    r"""Constrain `expr` to lie in the dual PPT cone.

    The PPT cone is \(\{M : M \succeq 0, \Gamma(M) \succeq 0\}\). Its dual cone is
    \(\text{Pos} + \Gamma(\text{Pos})\). We model this as `expr - partial_transpose(Q) >> 0`
    for an auxiliary positive semidefinite variable `Q`.

    """
    dim = expr.shape[0]
    q_var = picos.HermitianVariable(name, (dim, dim))
    problem.add_constraint(q_var >> 0)
    problem.add_constraint(expr - picos.partial_transpose(q_var, subsystems=subsystems, dimensions=dimensions) >> 0)


def _min_error_primal(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the primal problem for minimum-error quantum state exclusion SDP."""
    n = len(vectors)
    problem = picos.Problem()
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(n)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(picos.sum(measurements) == picos.I(dim))

    dms = [to_density_matrix(vector) for vector in vectors]

    problem.set_objective("min", np.real(picos.sum([(probs[i] * dms[i] | measurements[i]) for i in range(n)])))
    solution = problem.solve(solver=solver, **kwargs)
    return solution.value, measurements


def _min_error_dual(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the dual problem for minimum-error quantum state exclusion SDP."""
    n = len(vectors)
    problem = picos.Problem()

    # Set up variables and constraints for SDP:
    y_var = picos.HermitianVariable("Y", (dim, dim))
    problem.add_list_of_constraints([y_var << probs[i] * to_density_matrix(vector) for i, vector in enumerate(vectors)])

    # Objective function:
    problem.set_objective("max", picos.trace(y_var))
    solution = problem.solve(solver=solver, **kwargs)

    measurements = [problem.get_constraint(k).dual for k in range(n)]

    return solution.value, measurements


def _ppt_min_error_primal(
    vectors: list[np.ndarray],
    dim: int,
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the primal problem for minimum-error quantum state exclusion SDP with PPT constraints."""
    n = len(vectors)
    problem = picos.Problem()
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(n)]

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_list_of_constraints(
        [picos.partial_transpose(meas, subsystems=subsystems, dimensions=dimensions) >> 0 for meas in measurements]
    )
    problem.add_constraint(picos.sum(measurements) == picos.I(dim))

    dms = [to_density_matrix(vector) for vector in vectors]

    problem.set_objective("min", np.real(picos.sum([(probs[i] * dms[i] | measurements[i]) for i in range(n)])))
    solution = problem.solve(solver=solver, **kwargs)
    return solution.value, measurements


def _ppt_min_error_dual(
    vectors: list[np.ndarray],
    dim: int,
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Find the dual problem for minimum-error quantum state exclusion SDP with PPT constraints."""
    problem = picos.Problem()
    y_var = picos.HermitianVariable("Y", (dim, dim))

    for i, vector in enumerate(vectors):
        _ppt_dual_cone_constraint(
            problem=problem,
            expr=probs[i] * to_density_matrix(vector) - y_var,
            name=f"Q[{i}]",
            subsystems=subsystems,
            dimensions=dimensions,
        )

    problem.set_objective("max", picos.trace(y_var))
    solution = problem.solve(solver=solver, primals=None, **kwargs)
    measurements = [problem.get_constraint(2 * i + 1).dual for i in range(len(vectors))]

    return solution.value, measurements


def _unambiguous_primal(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Solve the primal problem for unambiguous quantum state distinguishability SDP.

    Implemented according to Equation (33) of [@bandyopadhyay2014conclusive].

    """
    n = len(vectors)
    problem = picos.Problem()
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(n)]
    inconclusive_measurement = picos.I(dim) - picos.sum(measurements)

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_constraint(inconclusive_measurement >> 0)

    unnormalized_dms = [p * to_density_matrix(vector) for (p, vector) in zip(probs, vectors)]
    sums_of_unnormalized_dms = picos.sum(unnormalized_dms)

    problem.add_list_of_constraints(m | rho == 0 for (m, rho) in zip(measurements, unnormalized_dms))

    problem.set_objective("min", picos.trace(sums_of_unnormalized_dms * inconclusive_measurement))
    solution = problem.solve(solver=solver, **kwargs)

    return solution.value, measurements + [inconclusive_measurement]


def _ppt_unambiguous_primal(
    vectors: list[np.ndarray],
    dim: int,
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, list[picos.HermitianVariable]]:
    """Solve the primal problem for unambiguous quantum state exclusion SDP with PPT constraints."""
    n = len(vectors)
    problem = picos.Problem()
    measurements = [picos.HermitianVariable(f"M[{i}]", (dim, dim)) for i in range(n)]
    inconclusive_measurement = picos.I(dim) - picos.sum(measurements)

    problem.add_list_of_constraints([meas >> 0 for meas in measurements])
    problem.add_list_of_constraints(
        [picos.partial_transpose(meas, subsystems=subsystems, dimensions=dimensions) >> 0 for meas in measurements]
    )
    problem.add_constraint(inconclusive_measurement >> 0)
    problem.add_constraint(
        picos.partial_transpose(inconclusive_measurement, subsystems=subsystems, dimensions=dimensions) >> 0
    )

    unnormalized_dms = [p * to_density_matrix(vector) for (p, vector) in zip(probs, vectors)]
    sum_of_unnormalized_dms = picos.sum(unnormalized_dms)

    problem.add_list_of_constraints(m | rho == 0 for (m, rho) in zip(measurements, unnormalized_dms))

    problem.set_objective("min", picos.trace(sum_of_unnormalized_dms * inconclusive_measurement))
    solution = problem.solve(solver=solver, **kwargs)

    return solution.value, measurements + [inconclusive_measurement]


def _unambiguous_dual(
    vectors: list[np.ndarray],
    dim: int,
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, tuple[picos.HermitianVariable, picos.RealVariable]]:
    """Solve the dual problem for unambiguous quantum state distinguishability SDP.

    Implemented according to Equation (35) of [@bandyopadhyay2014conclusive].

    """
    n = len(vectors)
    problem = picos.Problem()
    lagrangian_variable_big_n = picos.HermitianVariable("N", (dim, dim))
    lagrangian_variables_a = picos.RealVariable("a", n)

    problem.add_constraint(lagrangian_variable_big_n >> 0)

    dms = [to_density_matrix(vector) for (p, vector) in zip(probs, vectors)]
    unnormalized_dms = [proba * rho for (proba, rho) in zip(probs, dms)]
    sum_of_unnormalized_dms = picos.sum(unnormalized_dms)

    problem.add_list_of_constraints(
        (lagrangian_variable_big_n + lagrangian_variables_a[i] * unnormalized_dms[i] >> sum_of_unnormalized_dms)
        for i in range(n)
    )

    problem.set_objective("max", 1 - picos.trace(lagrangian_variable_big_n))
    solution = problem.solve(solver=solver, primals=None, **kwargs)

    return solution.value, (lagrangian_variable_big_n, lagrangian_variables_a)


def _ppt_unambiguous_dual(
    vectors: list[np.ndarray],
    dim: int,
    subsystems: list[int],
    dimensions: list[int],
    probs: list[float] | None = None,
    solver: str = "cvxopt",
    **kwargs,
) -> tuple[float, tuple[picos.HermitianVariable, picos.RealVariable]]:
    """Solve the dual problem for unambiguous quantum state exclusion SDP with PPT constraints."""
    n = len(vectors)
    problem = picos.Problem()
    y_var = picos.HermitianVariable("Y", (dim, dim))
    lagrangian_variables_a = picos.RealVariable("a", n)

    unnormalized_dms = [p * to_density_matrix(vector) for (p, vector) in zip(probs, vectors)]
    sum_of_unnormalized_dms = picos.sum(unnormalized_dms)

    _ppt_dual_cone_constraint(
        problem=problem,
        expr=sum_of_unnormalized_dms - y_var,
        name="Q[inconclusive]",
        subsystems=subsystems,
        dimensions=dimensions,
    )
    for i in range(n):
        _ppt_dual_cone_constraint(
            problem=problem,
            expr=lagrangian_variables_a[i] * unnormalized_dms[i] - y_var,
            name=f"Q[{i}]",
            subsystems=subsystems,
            dimensions=dimensions,
        )

    problem.set_objective("max", picos.trace(y_var))
    solution = problem.solve(solver=solver, primals=None, **kwargs)

    return solution.value, (y_var, lagrangian_variables_a)

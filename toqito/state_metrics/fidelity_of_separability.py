"""Add function for fidelity of separability as defined in [@philip2023schrodinger].

Fidelity of separability is an entanglement measure that can be approximated with semidefinite programs.
The constraints for this function are positive partial transpose (PPT) & k-extendible states.
"""

import numpy as np
import picos

from toqito.matrix_props import is_density
from toqito.perms import symmetric_projection
from toqito.state_props import is_pure, is_separable


def fidelity_of_separability(
    input_state_rho: np.ndarray,
    input_state_rho_dims: list[int],
    k: int = 1,
    verbosity_option: int = 0,
    solver_option: str = "cvxopt",
) -> float:
    r"""Define the first benchmark introduced in Appendix H of [@philip2023schrodinger].

    If you would like to instead use the benchmark introduced in Appendix I, go to
    [channel_metrics.fidelity_of_separability][toqito.channel_metrics.fidelity_of_separability].

    In [@philip2023schrodinger] a variational quantum algorithm (VQA) is introduced to test
    the separability of a general bipartite state. The algorithm utilizes
    quantum steering between two separated systems such that the separability
    of the state is quantified.

    Due to the limitations of currently available quantum computers, two
    optimization semidefinite programs (SDP) benchmarks were introduced to
    maximize the fidelity of separability subject to some state constraints
    (Positive Partial Transpose (PPT), symmetric extensions (k-extendibility
    ) [@hayden2013twomessage] ) This function approximites the fidelity of separability by
    maximizing over PPT states & k-extendible states i.e. an optimization
    problem over states [@watrous2018theory].

    The following expression (Equation (H2) from [@philip2023schrodinger] ) defines the
    constraints for approxiamting

    \(\sqrt{\widetilde{F}_s^1}(\rho_{AB}) {:}=\)

    \[
        \begin{multline}
        \max_{\substack{X_{AB} \in\mathcal{L}(\mathcal{H}_{AB}),\\\sigma_{AB^{k}}\geq0}}
        \left\{\begin{array}
                [c]{c}
                \operatorname{Re}[\operatorname{Tr}[X_{AB}]]:\\%
                \begin{bmatrix}
                \rho_{AB} & X_{AB}\\
                X_{AB}^{\dagger} & \sigma_{AB_{1}}%
                \end{bmatrix}
                \geq0,\\
                \operatorname{Tr}[\sigma_{AB^{k}}]=1,\\
                \sigma_{AB^{k}}=\mathcal{P}_{B^{k}}(\sigma_{AB^{k}}),\\
                T_{B_{1\cdots j}}(\sigma_{AB_{1\cdots j}})\geq 0 \quad \forall j\leq k
            \end{array}\right\}
        \end{multline}
    \]

    \(\sqrt{\widetilde{F}_s^1}(\rho_{AB})\) is the quantity to be
    approximated but this function returns
    \(\widetilde{F}_s^1(\rho_{AB})\).

    \(\operatorname{Re}[\operatorname{Tr}[X_{AB}]]\) is the maximization problem subject to PPT & k-extendibile
    state constraints.

    Here, \(\mathcal{L}(\mathcal{H}_{AB})\) is the space of linear operators over space \(\mathcal{H}_{AB}\).

    \(\sigma_{AB^{k}}\) is a k-extension of \(\rho_{AB}\).

    \(\mathcal{P}_{B^{k}}\) is the permutation operator among systems
    \(B_1, B_2,  \ldots , B_{k}\) which has no effect on the k-extended
    state \(\sigma_{AB^{k}}\).

    The other constraints are due to the PPT condition [@peres1996separability].

    Examples:
        Let's consider a density matrix of a state that we know is pure and separable; \(|00 \rangle = |0 \rangle
        \otimes |0 \rangle\).

        The expected approximation of fidelity of separability is the maximum value possible i.e. very close to 1.

        \[
            \rho_{AB} = |00 \rangle \langle 00|
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_metrics import fidelity_of_separability
        from toqito.matrix_ops import tensor
        from toqito.states import basis

        state = tensor(basis(2, 0), basis(2, 0))
        rho = state @ state.conj().T

        print(np.around(fidelity_of_separability(rho, [2, 2]), decimals=2))
        ```
            is PSD with trace 1).

    Raises:
        AssertionError: If the provided dimensions are not for a bipartite density matrix.
        ValueError: If the matrix is not a density matrix (square matrix that
        ValueError: the input state is entangled.
        ValueError: the input state is a mixed state.

    Args:
        input_state_rho: the density matrix for the bipartite state of interest.
        input_state_rho_dims: the dimensions of System A & B respectively in the input state density matrix. It is
            assumed that the first quantity in this list is the dimension of System A.
        k: value for k-extendibility.
        verbosity_option: Parameter option for `picos`. Default value is `verbosity = 0`. For more info, visit
            https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-verbosity.
        solver_option: Optimization option for `picos` solver. Default option is `solver_option="cvxopt"`. For more
            info, visit https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-solver.

    Returns:
        Optimized value of the SDP when maximized over a set of linear operators subject to some constraints.

    """
    # rho is relabelled as rho_{AB} where A >= B.
    if not is_density(input_state_rho):
        raise ValueError("Provided input state is not a density matrix.")
    if not len(input_state_rho_dims) == 2:
        raise AssertionError("For State SDP: require bipartite state dims.")
    if not is_pure(input_state_rho):
        raise ValueError("This function only works for pure states.")
    if not is_separable(input_state_rho):
        raise ValueError("Provided input state is entangled.")

    # Infer the dimension of Alice and Bob's system. subsystem-dimensions in rho_AB
    dim_a, dim_b = input_state_rho_dims

    # Extend the number of dimensions based on the level `k`. new dims for AB with k-extendibility in subsystem B
    dim_direct_sum_ab_k = [dim_a] + [dim_b] * (k)
    # new dims for a linear op acting on the space of sigma_ab_k
    dim_op_sigma_ab_k = dim_a * dim_b**k

    # A list of the symmetrically extended subsystems based on the level `k`.
    sub_sys_ext = list(range(2, 2 + k - 1))

    # unitary permutation operator in B1,B2,...,Bk
    permutation_op = symmetric_projection(dim_b, k)

    # defining the problem objective: Re[Tr[X_AB]]
    problem = picos.Problem(verbosity=verbosity_option)
    linear_op_ab = picos.ComplexVariable("x_ab", input_state_rho.shape)
    sigma_ab_k = picos.HermitianVariable("s_ab_k", (dim_op_sigma_ab_k, dim_op_sigma_ab_k))

    problem.set_objective("max", 0.5 * picos.trace(linear_op_ab + linear_op_ab.H))

    problem.add_constraint(
        picos.block(
            [
                [input_state_rho, linear_op_ab],
                [linear_op_ab.H, picos.partial_trace(sigma_ab_k, sub_sys_ext, dim_direct_sum_ab_k)],
            ]
        )
        >> 0
    )
    problem.add_constraint(sigma_ab_k >> 0)
    problem.add_constraint(picos.trace(sigma_ab_k) == 1)

    # k-extendible constraint:
    problem.add_constraint(
        (picos.I(dim_a) @ permutation_op) * sigma_ab_k * (picos.I(dim_a) @ permutation_op) == sigma_ab_k
    )

    # PPT constraint:
    sys = []
    for i in range(1, k):
        sys = sys + [i]
        problem.add_constraint(picos.partial_transpose(sigma_ab_k, sys, dim_direct_sum_ab_k) >> 0)

    solution = problem.solve(solver=solver_option)
    return solution.value**2

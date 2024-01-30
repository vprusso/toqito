"""Add function for fidelity of separability as defined in :cite:`Philip_2023_Schrodinger`.

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
    verbosity_option: int = 2,
    solver_option: str = "cvxopt",
) -> float:
    r"""Define the first benchmark introduced in Appendix H of :cite:`Philip_2023_Schrodinger`.

    If you would like to instead use the benchmark introduced in Appendix I, go to
    :obj:`toqito.channel_metrics.fidelity_of_separability`.

    In :cite:`Philip_2023_Schrodinger` a variational quantum algorithm (VQA) is introduced to test
    the separability of a general bipartite state. The algorithm utilizes
    quantum steering between two separated systems such that the separability
    of the state is quantified.

    Due to the limitations of currently available quantum computers, two
    optimization semidefinite programs (SDP) benchmarks were introduced to
    maximize the fidelity of separability subject to some state constraints
    (Positive Partial Transpose (PPT), symmetric extensions (k-extendibility
    ) :cite:`Hayden_2013_TwoMessage` ) This function approximites the fidelity of separability by
    maximizing over PPT states & k-extendible states i.e. an optimization
    problem over states :cite:`Watrous_2018_TQI`.

    The following expression (Equation (H2) from :cite:`Philip_2023_Schrodinger` ) defines the
    constraints for approxiamting

    :math:`\sqrt{\widetilde{F}_s^1}(\rho_{AB}) {:}=`

    .. math::

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

    :math:`\sqrt{\widetilde{F}_s^1}(\rho_{AB})` is the quantity to be
    approximated but this function returns
    :math:`\widetilde{F}_s^1(\rho_{AB})`.

    :math:`\operatorname{Re}[\operatorname{Tr}[X_{AB}]]` is the maximization problem subject to PPT & k-extendibile
    state constraints.

    Here, :math:`\mathcal{L}(\mathcal{H}_{AB})` is the space of linear operators over space :math:`\mathcal{H}_{AB}`.

    :math:`\sigma_{AB^{k}}` is a k-extension of :math:`\rho_{AB}`.

    :math:`\mathcal{P}_{B^{k}}` is the permutation operator among systems
    :math:`B_1, B_2,  \ldots , B_{k}` which has no effect on the k-extended
    state :math:`\sigma_{AB^{k}}`.

    The other constraints are due to the PPT condition :cite:`Peres_1996_Separability`.

    Examples
    ==========
    Let's consider a density matrix of a state that we know is pure and separable; :math:`|00 \rangle = |0 \rangle
    \otimes |0 \rangle`.

    The expected approximation of fidelity of separability is the maximum value possible i.e. very close to 1.

    .. math::
        \rho_{AB} = |00 \rangle \langle 00|

    .. code-block:: python

        from toqito.state_metrics import fidelity_of_separability
        from toqito.matrix_ops import tensor
        from toqito.states import basis

        state = tensor(basis(2, 0), basis(2, 0))
        rho = state @ state.conj().T
        expected_value = fidelity_of_separability(rho, [2, 2])
        expected_value

    >>> 0.9999999998278968

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param input_state_rho: the density matrix for the bipartite state of interest.
    :param input_state_rho_dims: the dimensions of System A & B respectively in
        the input state density matrix. It is assumed that the first
        quantity in this list is the dimension of System A.
    :param k: value for k-extendibility.
    :param verbosity_option: Parameter option for `picos`. Default value is
        `verbosity = 2`. For more info, visit
        https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-verbosity.
    :param solver_option: Optimization option for `picos` solver. Default option is
        `solver_option="cvxopt"`. For more info, visit
        https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-solver.
    :raises AssertionError: If the provided dimensions are not for a bipartite density matrix.
    :raises ValueError: If the matrix is not a density matrix (square matrix that
        is PSD with trace 1).
    :raises ValueError: the input state is entangled.
    :raises ValueError: the input state is a mixed state.
    :return: Optimized value of the SDP when maximized over a set of linear operators subject
        to some constraints.

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
    dim_op_sigma_ab_k = dim_a * dim_b ** k

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
    return solution.value ** 2

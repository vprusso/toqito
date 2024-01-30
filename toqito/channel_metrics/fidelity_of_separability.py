"""Add functions for channel fidelity of Separability as defined in :cite:`Philip_2023_Schrodinger`.

The constrainsts for this function are positive partial transpose (PPT)
& k-extendible channels.
"""

import numpy as np
import picos

from toqito.perms import permute_systems, symmetric_projection
from toqito.state_props import is_pure

from toqito.matrix_props import is_density  # isort: skip


def fidelity_of_separability(
    psi: np.ndarray,
    psi_dims: list[int],
    k: int = 1,
    verbosity_option: int = 2,
    solver_option: str = "cvxopt",
) -> float:
    r"""Define the first benchmark introduced in Appendix I of :cite:`Philip_2023_Schrodinger`.

    If you would like to instead use the benchmark introduced in Appendix H,
    go to :obj:`toqito.state_metrics.fidelity_of_separability`.

    In :cite:`Philip_2023_Schrodinger` a variational quantum algorithm (VQA) is introduced to test
    the separability of a general bipartite state. The algorithm utilizes
    quantum steering between two separated systems such that the separability
    of the state is quantified.

    Due to the limitations of currently available quantum computers, two
    optimization semidefinite programs (SDP) benchmarks were introduced to
    maximize the fidelity of separability subject to some state constraints (Positive Partial Transpose (PPT),
    symmetric extensions (k-extendibility) :cite:`Hayden_2013_TwoMessage`).
    Entangled states do not have k-symmetric extensions. If an extension exists, it cannot be assumed directly
    that the state is separable. This function approximites the fidelity of separability by
    maximizing over PPT channels & k-extendible entanglement breaking channels
    i.e. an optimization problem over channels :cite:`Watrous_2018_TQI` .

    The following discussion (Equation (I4) from :cite:`Philip_2023_Schrodinger` ) defines the
    constraints for approximating :math:`\widetilde{F}_s^2(\rho_{AB})` in
    :math:`\frac{1}{2}(1+\widetilde{F}_s^2(\rho_{AB}))`.

    .. math::
        \operatorname{Tr}[
            \Pi_{A^{\prime}A}^{\operatorname{sym}} \operatorname{Tr}_{R}[
                T_R(\psi_{RAB})\Gamma^{\mathcal{E}^{k}}_{RA^{\prime}_1}]]

    Above expression defines the maximization problem subject to PPT & k-extendibile channel
    constraints over :math:`\max_{\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}\geq 0}`

    The constraint expressions are listed below:

    .. math::
        \operatorname{Tr}_{A^{\prime k}}[\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}]=I_R

    :math:`\Gamma^{\mathcal{E}^{k}}_{RA^{\prime}}` is Choi operator of
    entanglement breaking channel :math:`\mathcal{E}^{k}`.

    .. math::
        \Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}= \mathcal{P}_{A^{\prime k}}(\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}})

    :math:`\mathcal{P}_{A^{\prime k}}` is the permutation operator over
    k-extensions :math:`A^{\prime k}`.

    .. math::
        T_{A^{\prime}_{1\cdots j}}(\Gamma^{\mathcal{E}^{k}_{RA^{\prime k}}}) \geq 0 \quad \forall j\leq k

    These other constraints are due to the PPT condition :cite:`Peres_1996_Separability`.

    Examples
    ==========
    Let's consider a density matrix of a state that we know is pure &
    separable. :math:`|000 \rangle = |0 \rangle \otimes |0 \rangle \otimes |0 \rangle`.

    The expected approximation of fidelity of separability is the maximum
    value possible i.e. very close to 1.

    .. math::
        \rho_{AB} = |000 \rangle \langle 000|

    .. code-block:: python

        from toqito.channel_metrics import fidelity_of_separability
        from toqito.matrix_ops import tensor
        from toqito.states import basis

        state = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
        rho = state*state.conj().T
        expected_value = fidelity_of_separability(rho, [2,2,2])

    >>> expected_value = 0.9999999979949119


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param psi: the density matrix for the tripartite state of interest psi_{BAR}
    :param psi_dims: the dimensions of System A, B, & R in
            the input state density matrix. It is assumed that the first
            quantity in this list is the dimension of System B.
    :param k: value for k-extendibility.
    :param verbosity_option: Parameter option for `picos`. Default value is
        `verbosity = 2`. For more info, visit
        https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-verbosity.
    :param solver_option: Optimization option for `picos` solver. Default option is
        `solver_option="cvxopt"`. For more info, visit
        https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-solver.
    :raises AssertionError: If the provided dimensions are not for a tripartite density matrix.
    :raises ValueError: If the matrix is not a density matrix (square matrix that
        is PSD with trace 1).
    :raises ValueError: the input state is entangled.
    :raises ValueError: the input state is a mixed state.
    :return: Optimized value of the SDP when maximized over a set of linear
        operators subject to some constraints.

    """
    if not is_density(psi):
        raise ValueError("Provided input state is not a density matrix.")
    tripartite_num = 3
    if not len(psi_dims) == tripartite_num:
        raise AssertionError("For Channel SDP: require tripartite state dims.")
    if not is_pure(psi):
        raise ValueError("This function only works for pure states.")

    # We first permure psi_{BAR} to psi_{RAB} to simplify the code.
    psi = permute_systems(psi, [3, 2, 1], psi_dims)
    dim_b, dim_a, dim_r = psi_dims
    psi_dims = [dim_r, dim_a, dim_b]

    # List of dimensions [R, A, B, A'] needed to define optimization function.
    dim_list = [dim_r, dim_a, dim_b, dim_a]

    # Dimension of the Choi matrix of the extended channel.
    choi_dims = [dim_r] + [dim_a] * k

    # List of extenstion systems and dimension of the Choi matrix.
    sys_ext = list(range(2, 2 + k - 1))
    dim_choi = dim_r * (dim_a ** k)

    # Projection onto symmetric subspace on AA'.
    pi_sym = symmetric_projection(dim_a, 2)

    choi = picos.HermitianVariable("S", (dim_choi, dim_choi))
    choi_partial = picos.partial_trace(choi, sys_ext, choi_dims)
    sym_choi = symmetric_projection(dim_a, k)
    problem = picos.Problem(verbosity=verbosity_option)

    problem.set_objective(
        "max",
        np.real(
            picos.trace(
                pi_sym
                * picos.partial_trace(
                    (picos.partial_transpose(psi, [0], psi_dims) @ picos.I(dim_a))
                    * permute_systems(
                        choi_partial @ picos.I(dim_b * dim_a), [1, 4, 3, 2], dim_list
                    ),
                    [0, 2],
                    dim_list,
                )
            )
        ),
    )

    problem.add_constraint(
        picos.partial_trace(choi, list(range(1, k + 1)), choi_dims) == picos.I(dim_r)
    )
    problem.add_constraint(choi >> 0)

    # k-extendablility of Choi state
    problem.add_constraint((picos.I(dim_r) @ sym_choi) * choi * (picos.I(dim_r) @ sym_choi) == choi)

    # PPT condition on Choi state
    sys = []
    for i in range(1, 1 + k):
        sys = sys + [i]
        problem.add_constraint(picos.partial_transpose(choi, sys, choi_dims) >> 0)

    solution = problem.solve(solver=solver_option)
    return 2 * solution.value - 1

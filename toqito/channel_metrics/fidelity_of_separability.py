"""Add functions for channel fidelity of Separability as defined in [@philip2023schrodinger].

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
    verbosity_option: int = 0,
    solver_option: str = "cvxopt",
) -> float:
    r"""Define the first benchmark introduced in Appendix I of [@philip2023schrodinger].

    If you would like to instead use the benchmark introduced in Appendix H, go to
    [state_metrics.fidelity_of_separability][toqito.state_metrics.fidelity_of_separability].

    In [@philip2023schrodinger], a variational quantum algorithm (VQA) is introduced to test the separability of
    general bipartite state. The algorithm utilizes quantum steering between two separated systems such that the
    separability of the state is quantified.

    Due to the limitations of currently available quantum computers, two optimization semidefinite programs (SDP)
    benchmarks were introduced to maximize the fidelity of separability subject to some state constraints
    (Positive Partial Transpose (PPT), symmetric extensions (k-extendibility) [@hayden2013twomessage]).
    Entangled states do not have k-symmetric extensions. If an extension exists, it cannot be assumed directly that the
    state is separable. This function approximates the fidelity of separability by maximizing over PPT channels &
    k-extendible entanglement breaking channels i.e. an optimization problem over channels [@watrous2018theory].

    The following discussion (Equation (I4) from [@philip2023schrodinger]) defines the constraints for approximating
    $\widetilde{F}_s^2(\rho_{AB})$ in $\frac{1}{2}(1+\widetilde{F}_s^2(\rho_{AB}))$.

    $$
    \operatorname{Tr}[
        \Pi_{A^{\prime}A}^{\operatorname{sym}} \operatorname{Tr}_{R}[
            T_R(\psi_{RAB})\Gamma^{\mathcal{E}^{k}}_{RA^{\prime}_1}]]
    $$

    Above expression defines the maximization problem subject to PPT & k-extendible channel constraints over
    $\max_{\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}\geq 0}$.

    The constraint expressions are listed below:

    $$
    \operatorname{Tr}_{A^{\prime k}}[\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}]=I_R
    $$

    $\Gamma^{\mathcal{E}^{k}}_{RA^{\prime}}$ is Choi operator of entanglement breaking channel $\mathcal{E}^{k}$.

    $$
    \Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}= \mathcal{P}_{A^{\prime k}}(\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}})
    $$

    $\mathcal{P}_{A^{\prime k}}$ is the permutation operator over k-extensions $A^{\prime k}$.

    $$
    T_{A^{\prime}_{1\cdots j}}(\Gamma^{\mathcal{E}^{k}_{RA^{\prime k}}}) \geq 0 \quad \forall j\leq k
    $$

    These other constraints are due to the PPT condition [@peres1996separability].

    Examples:
        Let's consider a density matrix of a state that we know is pure & separable.
        $|000 \rangle = |0 \rangle \otimes |0 \rangle \otimes |0 \rangle$.

        The expected approximation of fidelity of separability is the maximum
        value possible i.e. very close to 1.

        $$
        \rho_{AB} = |000 \rangle \langle 000|
        $$

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_metrics import fidelity_of_separability
        from toqito.matrix_ops import tensor
        from toqito.states import basis
        state = tensor(basis(2, 0), basis(2, 0))
        rho = state @ state.conj().T
        print(fidelity_of_separability(rho, [2, 2]))
        ```

    Raises:
        AssertionError: If the provided dimensions are not for a tripartite density matrix.
        ValueError: If the matrix is not a density matrix (square matrix that
            is PSD with trace 1).
        ValueError: the input state is entangled.
        ValueError: the input state is a mixed state.

    Args:
        psi: the density matrix for the tripartite state of interest psi_{BAR}
        psi_dims: the dimensions of System A, B, & R in the input state density matrix. It is assumed that the first
            quantity in this list is the dimension of System B.
        k: value for k-extendibility.
        verbosity_option: Parameter option for `picos`. Default value is
            `verbosity = 0`. For more info, visit
            https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-verbosity.
        solver_option: Optimization option for `picos` solver. Default option is
            `solver_option="cvxopt"`. For more info, visit
            https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-solver.

    Returns:
        Optimized value of the SDP when maximized over a set of linear operators subject to some constraints.

    """
    if not is_density(psi):
        raise ValueError("Provided input state is not a density matrix.")
    tripartite_num = 3
    if not len(psi_dims) == tripartite_num:
        raise AssertionError("For Channel SDP: require tripartite state dims.")
    if not is_pure(psi):
        raise ValueError("This function only works for pure states.")

    # We first permure psi_{BAR} to psi_{RAB} to simplify the code.
    psi = permute_systems(psi, [2, 1, 0], psi_dims)
    dim_b, dim_a, dim_r = psi_dims
    psi_dims = [dim_r, dim_a, dim_b]

    # List of dimensions [R, A, B, A'] needed to define optimization function.
    dim_list = [dim_r, dim_a, dim_b, dim_a]

    # Dimension of the Choi matrix of the extended channel.
    choi_dims = [dim_r] + [dim_a] * k

    # List of extenstion systems and dimension of the Choi matrix.
    sys_ext = list(range(2, 2 + k - 1))
    dim_choi = dim_r * (dim_a**k)

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
                    * permute_systems(choi_partial @ picos.I(dim_b * dim_a), [0, 3, 2, 1], dim_list),
                    [0, 2],
                    dim_list,
                )
            )
        ),
    )

    problem.add_constraint(picos.partial_trace(choi, list(range(1, k + 1)), choi_dims) == picos.I(dim_r))
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

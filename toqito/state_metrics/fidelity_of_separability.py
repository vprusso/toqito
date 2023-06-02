"""Add functions for fidelity of Separability as defined in.
Need to cite paper here.
"""

from toqito.state_props import is_pure, has_symmetric_extension
from toqito.matrix_props import is_density
import picos
import numpy as np
from toqito.perms import symmetric_projection
from toqito.perms import permute_systems


def state_ppt_extendible_fidelity(
    input_state_rho: np.ndarray, input_state_rho_dims: list[int], k: int = 1
) -> float:
    """Define the first benchmark introduced in Appendix H of.
    Examples
    ==========
    Add a detailed explanation here.
    Args:
        input_state_rho: the density matrix for the bipartite state of interest
        input_state_rho_dims: the dimensions of System A & B respectively in
            the input state density matrix. It is assumed that the first
            quantity in this list is the dimension of System A.
        k: value for k-extendibility
    Raises:
        AssertionError:
            * If the provided dimensions are not for a bipartite density matrix
        TypeError:
            * If the matrix is not a density matrix (square matrix that is
            * PSD with trace 1)
    Returns:
        Optimized value of the SDP when maximized over a set of linear
        operators subject to some constraints.
    """
    # rho is relabelled as rho_{AB} where A >= B.
    if not is_density(input_state_rho):
        raise ValueError("Provided input state is not a density matrix.")
    if not len(input_state_rho_dims) == 2:
        raise AssertionError("For State SDP: require bipartite state dims.")
    if not is_pure(input_state_rho):
        raise ValueError("This function only works for pure states.")
    if not has_symmetric_extension(input_state_rho):
        raise ValueError("Provided input state is entangled.")

    # Infer the dimension of Alice and Bob's system.
    # subsystem-dimensions in rho_AB
    dim_A, dim_B = input_state_rho_dims

    # Extend the number of dimensions based on the level `k`.
    # new dims for AB with k-extendibility in subsystem B
    dim_direct_sum_AB_k = [dim_A] + [dim_B] * (k)
    # new dims for a linear op acting on the space of sigma_AB_k
    dim_op_sigma_AB_k = dim_A * dim_B**k

    # A list of the symmetrically extended subsystems based on the level `k`.
    sub_sys_ext = list(range(2, 2 + k - 1))
    # #unitary permutation operator in B1,B2,...Bk
    permutation_op = symmetric_projection(dim_B, k)

    # defining the problem objective: Re[Tr[X_AB]]
    problem = picos.Problem(verbosity=2)
    linear_op_AB = picos.ComplexVariable("x_AB", input_state_rho.shape)
    sigma_AB_k = picos.HermitianVariable(
        "s_AB_k", (dim_op_sigma_AB_k, dim_op_sigma_AB_k))

    problem.set_objective("max", 0.5*picos.trace(
        linear_op_AB + linear_op_AB.H))

    # constraints
    # >>0 is enforcing positive semi-definite condition
    problem.add_constraint(picos.block([
        [input_state_rho, linear_op_AB],
        [linear_op_AB.H, picos.partial_trace(
            sigma_AB_k, sub_sys_ext, dim_direct_sum_AB_k)]
    ]) >> 0)
    problem.add_constraint(sigma_AB_k >> 0)
    problem.add_constraint(picos.trace(sigma_AB_k) == 1)

    # k-extendible:
    problem.add_constraint((
        picos.I(dim_A) @ permutation_op) * sigma_AB_k * (picos.I(
            dim_A) @ permutation_op) == sigma_AB_k)

    # PPT:
    sys = []
    for i in range(1, k):
        sys = sys+[i]
        problem.add_constraint(
            picos.partial_transpose(sigma_AB_k, sys, dim_direct_sum_AB_k) >> 0)

    solution = problem.solve(solver="cvxopt")
    return (solution.value)**2


def channel_ppt_extendible_fidelity(
    psi: np.ndarray, psi_dims: list[int], k: int = 1
) -> float:
    """Define the second benchmark introduced in Appendix I of.
    Examples
    ==========
    Add a detailed explanation here.
    Args:
        input_state_rho: the density matrix for the bipartite state of interest
        input_state_rho_dims: the dimensions of System A & B respectively in
            the input state density matrix.
        k: value for k-extendibility
    Raises:
        AssertionError:
            * If the provided dimensions are not for a bipartite density matrix
        TypeError:
            * If the matrix is not a density matrix (square matrix that is
            * PSD with trace 1)
    Returns:
        Optimized value of the SDP when maximized over a set of linear
        operators subject to some constraints.
    """
    if not is_density(psi):
        raise ValueError("Provided input state is not a density matrix.")
    if not len(psi_dims) == 3:
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
    dim_choi = dim_r * (dim_a**k)

    # Projection onto symmetric subspace on AA'.
    pi_sym = symmetric_projection(dim_a, 2)

    choi = picos.HermitianVariable("S", (dim_choi, dim_choi))
    choi_partial = picos.partial_trace(choi, sys_ext, choi_dims)
    sym_choi = symmetric_projection(dim_a, k)
    problem = picos.Problem(verbosity=2)

    problem.set_objective(
        "max",
        np.real(picos.trace(
            pi_sym *
            picos.partial_trace(
                (picos.partial_transpose(
                    psi, [0], psi_dims) @ picos.I(dim_a)) *
                permute_systems(
                    choi_partial @ picos.I(dim_b*dim_a), [
                        1, 4, 3, 2], dim_list),
                [0, 2], dim_list
            )
        ))
    )

    problem.add_constraint(
        picos.partial_trace(choi, list(range(
            1, k+1)), choi_dims) == picos.I(dim_r)
    )
    problem.add_constraint(choi >> 0)

    # k-extendablility of Choi state
    problem.add_constraint(
        (picos.I(dim_r) @ sym_choi) * choi * (picos.I(dim_r)@sym_choi) == choi
    )

    # PPT condition on Choi state
    sys = []
    for i in range(1, 1+k):
        sys = sys + [i]
        problem.add_constraint(
            picos.partial_transpose(choi, sys, choi_dims) >> 0)

    solution = problem.solve(solver="cvxopt")
    return 2 * solution.value - 1


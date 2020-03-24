"""Calculates probability of distinguishing via PPT measurements."""
from typing import List
import cvxpy
import numpy as np

from toqito.super_operators.partial_transpose import partial_transpose


def _partial_transpose_dense(rho, mask):
    """
    Based on Jonas' implementation using numpy.
    Very fast for dense problems.
    """
    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                            [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

    dim = [2] * len(mask) * 2
    data = rho.reshape(np.array(dim).flatten()).transpose(pt_idx).reshape(rho.shape)

    return data


def ppt_distinguishability(states: List[np.ndarray],
                           probs: List[float] = None) -> float:
    r"""
    Compute probability of distinguishing a state via PPT measurements.

    Implements the semidefinite program (SDP) whose optimal value is equal to
    the maximum probability of perfectly distinguishing orthogonal maximally
    entangled states using any PPT measurement; a measurement whose operators
    are positive under partial transpose. This SDP was explicitly provided in
    [1].

    References:
    [1] Cosentino, Alessandro.
        "Positive-partial-transpose-indistinguishable states via semidefinite
        programming."
        Physical Review A 87.1 (2013): 012321.
        https://arxiv.org/abs/1205.1031

    :param states: A list of density operators (matrices) corresponding to
                   quantum states.
    :param probs: A list of probabilities where `probs[i]` corresponds to the
                  probability that `states[i]` is selected by Alice.
    :return: The optimal probability with which the states can be distinguished
             via PPT measurements.
    """
    # Assume that at least one state is provided.
    if states is None or states == []:
        raise ValueError("InvalidStates: There must be at least one state "
                         "provided.")

    # Assume uniform probability if no specific distribution is given.
    if probs is None:
        probs = [1/len(states)] * len(states)
    if not np.isclose(sum(probs), 1):
        raise ValueError("Invalid: Probabilities must sum to 1.")

    dim_x, dim_y = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if dim_y == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    constraints = []
    y_var = cvxpy.Variable((dim_x, dim_x), hermitian=True)
    objective = 1/len(states) * cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    dim = int(np.log2(dim_x))
    dim_list = [2] * int(np.log2(dim_x))
    sys_list = list(range(1, dim, 2))

    for i, _ in enumerate(states):
        constraints.append(cvxpy.real(y_var) >> partial_transpose(states[i],
                                                                  sys=sys_list,
                                                                  dim=dim_list))

    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default

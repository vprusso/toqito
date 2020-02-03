import numpy as np
from cvxpy.expressions.expression import Expression
from toqito.helper.cvx_helper import expr_as_np_array, np_array_as_expr
from numpy import linalg as LA
import cvxpy
from toqito.helper.constants import e0, e1, e00, e01, e10, e11, ep, em
from toqito.matrix.operations.tensor import tensor_list
from collections import defaultdict
from toqito.states.w_state import w_state
from toqito.states.bell import bell
from toqito.perms.unique_perms import unique_perms
from toqito.matrix.operations.vec import vec
from toqito.helper.iden import iden
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
from toqito.states.ghz_state import ghz_state
from toqito.hedging.pi_perm import pi_perm
from typing import List
from scipy.sparse import issparse
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.hedging.calculate_hedging_value import calculate_hedging_value
from toqito.states.pure_to_mixed import pure_to_mixed
from toqito.states.purity import purity
from numpy.linalg import matrix_power
from toqito.super_operators.choi_map import choi_map
from toqito.super_operators.reduction_map import reduction_map
from toqito.super_operators.partial_trace import partial_trace, partial_trace_cvx
from toqito.super_operators.apply_map import apply_map
from toqito.super_operators.partial_transpose import partial_transpose
from toqito.super_operators.dephasing_channel import dephasing_channel
from toqito.super_operators.depolarizing_channel import depolarizing_channel
from toqito.matrix.matrices.fourier_matrix import fourier_matrix
from toqito.super_operators.partial_map import partial_map
from toqito.states.state_exclusion import state_exclusion
from toqito.hedging.weak_coin_flipping import weak_coin_flipping
from toqito.super_operators.realignment import realignment
from toqito.states.chessboard_state import chessboard_state
from toqito.states.horodecki_state import horodecki_state
from toqito.matrix.matrices.gell_mann import gell_mann
from toqito.perms.permutation_operator import permutation_operator
from toqito.entanglement.negativity import negativity
import scipy as sp
from toqito.states.entropy import entropy
from toqito.states.schmidt_decomposition import schmidt_decomposition
from toqito.states.max_entangled import max_entangled

n = 2
k = 1
alpha = 1/np.sqrt(2)
theta = np.pi/8

v = np.cos(theta)*e00 + np.sin(theta)*e11
P1 = v * v.conj().T
P0 = np.identity(4) - P1

w = alpha * np.cos(theta)*e00 + np.sqrt(1-alpha**2)*np.sin(theta)*e11
l1 = -alpha*np.sin(theta)*e00 + np.sqrt(1-alpha**2)*np.cos(theta)*e11
l2 = alpha*np.sin(theta)*e10
l3 = np.sqrt(1-alpha**2)*np.cos(theta)*e01
 
Q1 = w * w.conj().T
Q0 = l1 * l1.conj().T + l2 * l2.conj().T + l3 * l3.conj().T

u = 1/np.sqrt(2) * (e00 + e11)
rho = u * u.conj().T

test_input_mat = np.array([[1, 4, 7],
                           [2, 5, 8],
                           [3, 6, 9]])

expected_res = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

test_input_mat = np.array([[1, 2],
                           [3, 4]])

schmidt_decomposition(max_entangled(3))


#print(entropy(rho))
#print(entropy(rho, 10))
#print(entropy(np.identity(4)/4))
#print(entropy(np.identity(4)/4))
#print(entropy(rho, 2, -1))

#print(iden([2, 2], is_sparse=False))
#kraus_1 = np.array([[1, 5], [1, 0], [0, 2]])
#res = apply_map(test_input_mat, kraus_1)
#print(res)


#print(isinstance(iden(4, is_sparse=True), sp.sparse.dia_matrix))
#print(permutation_operator(2, 2))
#print(permutation_operator([2, 2], [2, 1]))
#print(permutation_operator(3, [2, 1], False, True))


# Q11 = np.kron(Q1, Q1)
#X = cvxpy.Variable((4, 4), PSD=True)
#objective = cvxpy.Maximize(cvxpy.trace(Q0 * X))
#constraints = [partial_trace_cvx(X, sys=1, dim=[2]) == np.identity(2)]
#problem = cvxpy.Problem(objective, constraints)
#sol_default = problem.solve()

# #print(partial_trace(Q11, sys=[1, 2]))
# 
#print(sol_default)
#print(np.around(X.value, decimals=4))

# sdp_vars = []
# obj_func = []
# constraints = []
# 
# M0 = cvx.Variable((4, 4), PSD=True)
# M1 = cvx.Variable((4, 4), PSD=True)
# M2 = cvx.Variable((4, 4), PSD=True)
# measurements = [M0, M1, M2]
# 
# obj_func.append(probs[0] * cvx.trace(states[0].conj().T * measurements[0]))
# obj_func.append(probs[1] * cvx.trace(states[1].conj().T * measurements[1]))
# obj_func.append(probs[2] * cvx.trace(states[2].conj().T * measurements[2]))
# 
# constraints.append(M0 + M1 + M2 == np.identity(4))
#         
# objective = cvx.Maximize(sum(obj_func))
# problem = cvx.Problem(objective, constraints)
# sol_default = problem.solve()
# print(1/len(states) * sol_default)

#     state = np.kron(e1*e1.conj().T, e0*e0.conj().T) + np.kron(em*em.conj().T, e1*e1.conj().T)
# 
#     rho1_AM = cvx.Variable((4, 4), PSD=True)
#     objective = cvx.Maximize(cvx.trace(state * rho1_AM))
#     constraints = [partial_trace(rho1_AM, dims=[2, 2], axis=1) == 1/2 * np.identity(2)]
#     problem = cvx.Problem(objective, constraints)
#     sol_default = problem.solve()
# 
#     print(sol_default)
#     print(np.around(rho1_AM.value, decimals=4))
# 

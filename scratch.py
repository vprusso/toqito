import numpy as np
from numpy import linalg as LA
import cvxpy as cvx
import itertools
from toqito.helper.constants import e0, e1, e00, e11
from toqito.matrix.operations.tensor import tensor_list
from collections import defaultdict
from toqito.states.w_state import w_state
from toqito.states.bell import bell
from toqito.perms.unique_perms import unique_perms
from toqito.matrix.operations.vec import vec
from toqito.helper.iden import iden
from scipy.sparse import csr_matrix, lil_matrix
import operator
import functools
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
from toqito.super_operators.partial_trace import partial_trace
from toqito.super_operators.apply_map import apply_map
from toqito.super_operators.partial_transpose import partial_transpose
from toqito.super_operators.dephasing_channel import dephasing_channel
from toqito.super_operators.depolarizing_channel import depolarizing_channel
from toqito.matrix.matrices.fourier_matrix import fourier_matrix
from toqito.super_operators.partial_map import partial_map
from toqito.states.state_exclusion import state_exclusion

n = 2
k = 1
alpha = 1/np.sqrt(2)
theta = np.pi/8

v = np.cos(theta)*e00 + np.sin(theta)*e11
P1 = v * v.conj().T
P0 = np.identity(4) - P1


X = np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
Phi = choi_map(3)
#print(apply_map(X, Phi))
#print(partial_map(X, Phi))

rho0 = bell(0) * bell(0).conj().T
rho1 = bell(1) * bell(1).conj().T
rho2 = bell(2) * bell(2).conj().T
states = [rho0, rho1, rho2]
probs = [1/3, 1/3, 1/3]

print(state_exclusion(states))

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

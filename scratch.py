import numpy as np
from toqito.hedging.pi_perm import pi_perm
from toqito.hedging.hedging_sdps import maximize_losing_less_than_k
from toqito.matrix.properties.is_normal import is_normal
from toqito.random.random_state_vector import random_state_vector
from toqito.base.ket import ket
from toqito.state.distance.trace_norm import trace_norm
from toqito.state.distance.trace_distance import trace_distance
from toqito.state.properties.is_product_vector import is_product_vector
from toqito.state.operations.schmidt_decomposition import schmidt_decomposition
from toqito.state.states.max_entangled import max_entangled
from toqito.matrix.matrices.pauli import pauli
from toqito.entanglement.concurrence import concurrence
import scipy

n = 2
k = 1
alpha = 1/np.sqrt(2)
theta = np.pi/8

e0, e1 = ket(2, 0), ket(2, 1)
e00 = np.kron(e0, e0)
e01 = np.kron(e0, e1)
e10 = np.kron(e1, e0)
e11 = np.kron(e1, e1)

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

Q11 = np.kron(Q1, Q1)
x = maximize_losing_less_than_k(Q1, n)

a = pi_perm(n-1)
b = np.kron(np.identity(2**(n-1)), np.identity(2**(n-1)))
c = pi_perm(n-1).conj().T

u = a * b * c

v = random_state_vector(30)
print(is_product_vector(v, [2, 3, 5]))
#v = np.array([1, 0, 0, 0, 1, 0, 0, 0])/np.sqrt(2)
#print(is_product_vector(v, [2, 2, 2]))

#print(concurrence(rho))

x = pauli("i", True)
print(scipy.sparse.issparse(x))

#print(trace_norm(rho))


#u, s, vh = np.linalg.svd(rho)

#print(np.sum(s))
#print(np.linalg.norm(rho, ord="nuc"))

#print(np.sum(random_state_vector(2)))

# This is the trace norm
#print(np.sum(np.linalg.svd(rho)))

#X = cvxpy.Variable((4**(n-1), 4**(n-1)), PSD=True)
#objective = cvxpy.Maximize(cvxpy.trace(Q1.conj().T * X))
#constraints = [partial_trace_cvx(X, 1, [2, 2]) == np.identity(2**(n-1))]
#problem = cvxpy.Problem(objective, constraints)
#sol_default = problem.solve()
#print(sol_default)

#res = permute_systems(test_input_mat, [2, 1])
#print(res)

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

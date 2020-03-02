import numpy as np
import itertools
import cvxpy
import sys
from toqito.base.ket import ket
from numpy import kron, sqrt, cos, sin
from toqito.nonlocal_games.hedging.hedging_value import HedgingValue
from toqito.random.random_state_vector import random_state_vector
from toqito.state.operations.schmidt_rank import schmidt_rank
from toqito.matrix.operations.tensor import tensor_list
from toqito.perms.antisymmetric_projection import antisymmetric_projection
from toqito.super_operators.partial_trace import partial_trace_cvx, partial_trace
from toqito.perms.pi_perm import pi_perm
from toqito.perms.permutation_operator import permutation_operator
np.set_printoptions(threshold=sys.maxsize)


e0, e1 = ket(2, 0), ket(2, 1)
ep = (e0 + e1)/np.sqrt(2)
em = (e0 - e1)/np.sqrt(2)

states = [e0, e1, ep, em]
probs = [1/4, 1/4, 1/4, 1/4]
dim = states[0].shape[0]**3


Q = np.zeros((dim, dim))
for i in range(len(states)):
    Q += probs[i] * tensor_list([states[i], states[i], states[i].conj()])*tensor_list([states[i], states[i], states[i].conj()]).conj().T

sys = [1, 2]
dims = [2, 2, 2]

x_var = cvxpy.Variable((dim, dim), PSD=True)
objective = cvxpy.Maximize(cvxpy.trace(Q.conj().T @ x_var))
constraints = [
    partial_trace_cvx(x_var, sys, dims) == np.identity(2)
]
problem = cvxpy.Problem(objective, constraints)

print(problem.solve())

q00 = kron(e0, e0)
q01 = kron(e0, e1)
q02 = kron(e0, ep)
q03 = kron(e0, em)

q10 = kron(e1, e0)
q11 = kron(e1, e1)
q12 = kron(e1, ep)
q13 = kron(e1, em)

q20 = kron(ep, e0)
q21 = kron(ep, e1)
q22 = kron(ep, ep)
q23 = kron(ep, em)

q30 = kron(em, e0)
q31 = kron(em, e1)
q32 = kron(em, ep)
q33 = kron(em, em)

qb00 = q00 * q00.conj().T
qb01 = q01 * q01.conj().T
qb02 = q02 * q02.conj().T
qb03 = q03 * q03.conj().T

qb10 = q10 * q10.conj().T
qb11 = q11 * q11.conj().T
qb12 = q12 * q12.conj().T
qb13 = q13 * q13.conj().T

qb20 = q20 * q20.conj().T
qb21 = q21 * q21.conj().T
qb22 = q22 * q22.conj().T
qb23 = q23 * q23.conj().T

qb30 = q30 * q30.conj().T
qb31 = q31 * q31.conj().T
qb32 = q32 * q32.conj().T
qb33 = q33 * q33.conj().T

C = 1/16 * np.kron(np.kron(qb00, qb00), qb00) + 1/16 * np.kron(np.kron(qb01, qb01), qb01) + \
    1/16 * np.kron(np.kron(qb02, qb02), qb02) + 1/16 * np.kron(np.kron(qb03, qb03), qb03) + \
    1/16 * np.kron(np.kron(qb10, qb10), qb10) + 1/16 * np.kron(np.kron(qb11, qb11), qb11) + \
    1/16 * np.kron(np.kron(qb12, qb12), qb12) + 1/16 * np.kron(np.kron(qb13, qb13), qb13) + \
    1/16 * np.kron(np.kron(qb20, qb20), qb20) + 1/16 * np.kron(np.kron(qb21, qb21), qb21) + \
    1/16 * np.kron(np.kron(qb22, qb22), qb22) + 1/16 * np.kron(np.kron(qb23, qb23), qb23) + \
    1/16 * np.kron(np.kron(qb30, qb30), qb30) + 1/16 * np.kron(np.kron(qb31, qb31), qb31) + \
    1/16 * np.kron(np.kron(qb32, qb32), qb32) + 1/16 * np.kron(np.kron(qb33, qb33), qb33)


#p = permutation_operator(2, [1, 4, 2, 5, 3, 6])
#p = permutation_operator(4, [1, 2, 3])

# dim = 2
# l_1 = list(range(1, dim+1))
# l_2 = list(range(dim+1, dim**2+1))
# if dim == 1:
#     perm = [1]
# else:
#     perm = [*sum(zip(l_1, l_2), ())]
Q2 = kron(Q, Q)
#print(Q2[0])
#print(np.around(C[0], decimals=8))
#print(permutation_operator(2, [1, 2, 3, 4]).shape)
# print(perm)
# perms = list(itertools.permutations([1, 2, 3, 4, 5, 6]))
# for i in range(len(perms)):
#     p = permutation_operator(2, list(perms[i]))
#     Q2 = p * np.kron(Q, Q) * p.conj().T
#
#     if np.allclose(Q2, C, atol=0.01):
#         print(perms[i])
#         break

#0.3125
#0.09765625

# num_reps = 2
# sys = [1, 2]
# dims = num_reps * np.array([2, 2, 2])
# x_var = cvxpy.Variable((dim**num_reps, dim**num_reps), PSD=True)
# objective = cvxpy.Maximize(cvxpy.trace(Q2.conj().T @ x_var))
# constraints = [
#     partial_trace_cvx(x_var, sys, dims) == np.identity(2*num_reps)
# ]
# problem = cvxpy.Problem(objective, constraints)
# print(problem.solve())

d = 2
ia, ib, ic = 2, 2, 2
oa, ob, oc = 2, 2, 2

V = np.zeros((oa, ob, oc, ia, ib, ic))
p = np.zeros((ia, ib, ic))
for a in range(oa):
    for b in range(ob):
        for c in range(oc):
            for x in range(ia):
                for y in range(ib):
                    for z in range(ic):
                        if (x or y or z) == (a ^ b ^ c):
                            V[a, b, c, x, y, z] = 1

p[0, 0, 0] = 1/4
p[0, 1, 1] = 1/4
p[1, 0, 1] = 1/4
p[1, 1, 0] = 1/4

V = np.zeros((oa, ob, oc, ia, ib, ic))
p = np.zeros((ia, ib, ic))
for a in range(oa):
    for b in range(ob):
        for c in range(oc):
            for x in range(ia):
                for y in range(ib):
                    for z in range(ic):
                        if (x and y and z) == (a ^ b ^ c):
                            V[a, b, c, x, y, z] = 1
p[0, 0, 0] = 1/8
p[0, 0, 1] = 1/8
p[0, 1, 0] = 1/8
p[0, 1, 1] = 1/8
p[1, 0, 0] = 1/8
p[1, 0, 1] = 1/8
p[1, 1, 0] = 1/8
p[1, 1, 1] = 1/8

#three_player_quantum_lower_bound(d, p, V)


prob_mat = np.array([[1/4, 1/4],
                     [1/4, 1/4]])
pred_mat = np.array([[0, 0],
                     [0, 1]])

d = 2
oa, ob, ia, ib = 2, 2, 2, 2
p = np.array([[1/4, 1/4], [1/4, 1/4]])
V = np.zeros((oa, ob, ia, ib))

for a in range(oa):
    for b in range(ob):
        for x in range(ia):
            for y in range(ib):
                if np.mod(a+b+x*y, d) == 0:
                    V[a, b, x, y] = 1

#v = nonlocal_game_lower_bound(d, p, V)
#print(np.isclose(v, np.cos(np.pi/8)**2))

prob_mat = np.array([[1/4, 1/4],
                     [1/4, 1/4]])
pred_mat = np.array([[0, 0],
                     [0, 1]])
#print(xor_game_value(prob_mat, pred_mat, tol=1))


n = 1
k = 1
alpha = 1/np.sqrt(2)
theta = np.pi/8

e0, e1 = ket(2, 0), ket(2, 1)
e00 = np.kron(e0, e0)
e01 = np.kron(e0, e1)
e10 = np.kron(e1, e0)
e11 = np.kron(e1, e1)
ep = (e0 + e1)/np.sqrt(2)
em = (e0 - e1)/np.sqrt(2)

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

Q000 = tensor_list([Q0, Q1, Q0])
hv = HedgingValue(Q0, 1)
#print(pi_perm(3) == permutation_operator(2, [1, 4, 2, 5, 3, 6]))
print(hv.max_prob_outcome_a_dual())
print(hv.min_prob_outcome_a_dual())

#print(min_prob_outcome_a_primal(Q1, 1))
#print(min_prob_outcome_a_dual(Q1, 1))


#X = np.arange(1, 211).reshape(15, 14)
#mat = np.array([[5, 3], [2, 7]])
#print(realignment(X, mat))


#calculate_q(Q0, Q1, n, k)

#x = np.kron(e1*e1.conj().T, e0*e0.conj().T) + np.kron(em*em.conj().T, e1*e1.conj().T)
#print(weak_coin_flipping(x))

#X = np.arange(1, 17).reshape(4, 4)
#print(np.linalg.norm(partial_transpose(X, [1, 2]) - X.conj().T))
#print(partial_trace(X, 2, [2, 2]))
#print(partial_transpose(X, [1, 2], [2, 2]))

#X = np.arange(1, 257).reshape(16, 16)
#Y = partial_transpose(X, [1, 3], [2, 2, 2, 2])
#print(partial_trace(X, [1, 3], [2, 2, 2, 2]))

#Q1 = np.kron(Q1, np.kron(Q1, Q1))
#Q1 = np.kron(Q1, Q1)
#print(partial_trace(Q1, [1, 3, 5], [2, 2, 2, 2, 2, 2]))
#print(partial_trace(Q1, [1, 3, 5, 7], [2, 2, 2, 2, 2, 2, 2, 2]))
#maximize_losing_less_than_k(Q1, n)
#minimize_losing_less_than_k(Q0, n)
#minimize_losing_less_than_k(Q00, n=2)


#v1 = [1/np.sqrt(2), 0, -1/np.sqrt(2)]
#print(np.linalg.norm(v1))


#v1 /= np.linalg.norm(v1)
#print(v1)


#Z = cvxpy.Variable(rho.shape, complex=True)
#objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(Z + Z.H)))
#constraints = [cvxpy.bmat([[rho, Z], [Z.H, sigma]]) >> 0]
#problem = cvxpy.Problem(objective, constraints)

#print(is_ppt(np.identity(9)))
#print(is_ppt(rho))

#X = cvxpy.Variable((4**(n-1), 4**(n-1)), PSD=True)
#objective = cvxpy.Maximize(cvxpy.trace(Q_a.conj().T @ X))
#constraints = [partial_trace_cvx(X, sys, dim) == np.identity(2**(n-1))]
#problem = cvxpy.Problem(objective, constraints)

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

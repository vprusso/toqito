import numpy as np
from toqito.base.ket import ket
from toqito.nonlocal_games.hedging.hedging_sdps import min_prob_outcome_a_dual, min_prob_outcome_a_primal


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


# d = 3
# oa, ob, ia, ib = 3, 3, 3, 3
# p = np.ones((d, d))/d**2
# V = np.zeros((oa, ob, ia, ib))
# for a in range(oa):
#     for b in range(ob):
#         for x in range(ia):
#             for y in range(ib):
#                 if np.mod(a+b+x*y, d) == 0:
#                     V[a, b, x, y] = 1
# 
# print(nonlocal_game_value_lb(d, p, V))


joint_coe = np.array([[1, 1, -1], [1, 1, 1], [-1, 1, 0]])
a_coe = np.array([0, -1, 0])
b_coe = np.array([-1, -2, 0])
a_val = np.array([0, 1])
b_val = np.array([0, 1])
#bell_inequality_max_qubits(joint_coe, a_coe, b_coe, a_val, b_val)


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

k = 1
n = 1

#Q0 = np.kron(Q0, Q0)
print(min_prob_outcome_a_primal(Q1, 1))
print(min_prob_outcome_a_dual(Q1, 1))


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

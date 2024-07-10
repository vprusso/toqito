import numpy as np

from toqito.matrices import standard_basis
from toqito.nonlocal_games import ExtendedNonlocalGame
from toqito.rand import random_povm
from toqito.states import trine

import numpy as np
from toqito.matrices import standard_basis
from toqito.state_props import concurrence
e_0, e_1 = standard_basis(2)
e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)
u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
rho = u_vec @ u_vec.conj().T
print(np.around(concurrence(rho), decimals=2))

eps = 0.5
print(np.around(1/3 * (2 + np.sqrt(1 - eps**2)), decimals=3))


exit()

e_0, e_1 = standard_basis(2)
ep = (e_0 + e_1) / np.sqrt(2)
em = (e_0 - e_1) / np.sqrt(2)

dim = 2
num_alice_out, num_bob_out = 2, 2
num_alice_in, num_bob_in = 3, 3

povms = random_povm(dim=dim, num_inputs=num_alice_in, num_outputs=num_alice_out)

print(povms[:,:,0,0] + povms[:,:,0,1])
print(ep @ ep.conj().T + em @ em.conj().T)
exit()

pred_mat = np.zeros([dim, dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in])
pred_mat[:, :, 0, 0, 0, 0] = povms[:, :, 0, 0]
pred_mat[:, :, 0, 0, 1, 1] = povms[:, :, 1, 0]
pred_mat[:, :, 0, 0, 2, 2] = povms[:, :, 2, 0]

pred_mat[:, :, 1, 1, 0, 0] = povms[:, :, 0, 1]
pred_mat[:, :, 1, 1, 1, 1] = povms[:, :, 1, 1]
pred_mat[:, :, 1, 1, 2, 2] = povms[:, :, 2, 1]

prob_mat = 1 / 3 * np.identity(3)

game = ExtendedNonlocalGame(prob_mat, pred_mat, reps=1)
unent = game.unentangled_value()
print(f"{unent=}")
lb = game.quantum_value_lower_bound()
print(f"{lb=}")
ns = game.nonsignaling_value()
print(f"{ns=}")

exit()

e_0, e_1 = standard_basis(2)
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

dim = 2
num_alice_out, num_bob_out = 2, 2
num_alice_in, num_bob_in = 2, 2

pred_mat = np.zeros([dim, dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in])

#pred_mat[:, :, 0, 1, 0, 0] = e_0 @ e_0.conj().T
#pred_mat[:, :, 1, 0, 0, 0] = e_0 @ e_0.conj().T
pred_mat[:, :, 1, 1, 0, 0] = e_0 @ e_0.conj().T

pred_mat[:, :, 0, 0, 0, 1] = e_1 @ e_1.conj().T
#pred_mat[:, :, 1, 0, 0, 1] = e_1 @ e_1.conj().T
pred_mat[:, :, 1, 1, 0, 1] = e_1 @ e_1.conj().T

pred_mat[:, :, 0, 0, 1, 0] = e_p @ e_p.conj().T
#pred_mat[:, :, 0, 1, 1, 0] = e_p @ e_p.conj().T
pred_mat[:, :, 1, 1, 1, 0] = e_p @ e_p.conj().T

pred_mat[:, :, 0, 0, 1, 1] = e_m @ e_m.conj().T
#pred_mat[:, :, 0, 1, 1, 1] = e_m @ e_m.conj().T
#pred_mat[:, :, 1, 0, 1, 1] = e_m @ e_m.conj().T

prob_mat = 1 / 2 * np.identity(2)
#prob_mat = np.array([[1/4, 1/4], [1/4, 1/4]])

game = ExtendedNonlocalGame(prob_mat, pred_mat, reps=2)
res = game.unentangled_value()
#res = game.nonsignaling_value()
#res = game.quantum_value_lower_bound()
print(res)

#res = game.nonsignaling_value()
#res = game.commuting_measurement_value_upper_bound()
#res = game.quantum_value_lower_bound()

exit()
dim = 2
num_in = 3
num_out = 2
pred_mat = np.zeros([dim, dim, num_out, num_out, num_in, num_in], dtype=complex)

e0, e1 = standard_basis(2)

theta_1 = 1/4 * np.arccos((121 + 52 * np.sqrt(13))/(477))
theta_2 = 1/4 * np.arccos((-431 + 4 * np.sqrt(13))/(477))
alpha_0 = -theta_1
alpha_1 = theta_2
beta_0 = (np.pi/2) - theta_2
beta_1 = theta_1
def M(theta):
    return np.array([
        [np.cos(theta)**2, np.cos(theta) * np.sin(theta)],
        [np.sin(theta) * np.cos(theta), np.sin(theta)**2]
    ])

# V(0,0|x,y):
pred_mat[:, :, 0, 0, 1, 1] = M(alpha_0)
pred_mat[:, :, 0, 0, 2, 2] = np.identity(dim) - M(alpha_0)

# V(0,1|x,y):
pred_mat[:, :, 0, 1, 0, 0] = M(alpha_1)
pred_mat[:, :, 0, 1, 1, 1] = np.identity(dim) - M(alpha_1)

# V(1,0|x,y):
pred_mat[:, :, 1, 0, 0, 0] = M(beta_0)
pred_mat[:, :, 1, 0, 1, 1] = np.identity(dim) - M(beta_0)

# V(1,1|x,y):
pred_mat[:, :, 1, 1, 0, 0] = M(beta_1)
pred_mat[:, :, 1, 1, 2, 2] = np.identity(dim) - M(beta_1)

prob_mat = 1 / 3 * np.identity(3)

game = ExtendedNonlocalGame(prob_mat, pred_mat)
res = game.unentangled_value()
#res = game.nonsignaling_value()
#res = game.commuting_measurement_value_upper_bound()
#res = game.quantum_value_lower_bound()
print(res)

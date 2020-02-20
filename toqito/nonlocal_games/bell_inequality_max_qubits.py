import numpy as np


def bell_inequality_max_qubits(joint_coe, a_coe, b_coe, a_val, b_val):
    ma, mb = joint_coe.shape[0], joint_coe.shape[1]
    oa = len(a_val)
    ob = len(b_val)

    # Turn row vectors into column vectors.
    a_val = a_val.reshape(-1, 1)
    b_val = b_val.reshape(-1, 1)

    a_coe = a_coe.reshape(-1, 1)
    b_coe = b_coe.reshape(-1, 1)

    if len(a_val != 2) or len(b_val) != 2:
        raise ValueError("Invalid: This function os only capable of handling Bell inequalities with two outcomes.")
    m = ma

    tot_dim = 2**(2*m+2)
    obj_mat = np.zeros((tot_dim, tot_dim))

    for a in range(1):
        for b in range(1):
            for x in range(1, m):
                for y in range(1, m):
                    b_coeff = joint_coe[x, y] * a_val[a] * b_val[b]
                    if y == 1:
                        b_coeff = b_coeff + a_coe[x] * a_val[a]
                    if x == 1:
                        b_coeff = b_coeff + b_coe[y] * b_val[b]
             #       obj_mat = obj_mat + b_coeff*np.kron(mn_matrix(m, a, x), mn_matrix(m, a, x))
#    obj_mat = 


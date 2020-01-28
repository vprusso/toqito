import numpy as np
from toqito.perms.unique_perms import unique_perms


def calculate_q(Q0, Q1, n, k):
    v = np.ones(n)
    perms = v
    for i in range(k-1):
        v[k-i] = 2
        perms = np.array([[perms], [unique_perms(v)]])

    for i in range(1):
        for j in range(len(perms)):
            print(perms[0, 0])
        

    return Q0, Q1

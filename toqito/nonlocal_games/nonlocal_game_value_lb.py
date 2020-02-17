import numpy as np
from toqito.random.random_density_matrix import random_density_matrix


def nonlocal_game_value_lb(d, p, V):

    # Get some basic values.
    ma, mb = p.shape
    oa, ob = V.shape[0], V.shape[1]

    # Generate random starting measurements for Bob.
    B = np.zeros((d, d, ob, mb), order="F", dtype=complex)
    for y in range(mb):
        sum_B = np.zeros(d)
        for b in range(ob):
            B[:, :, b, y] = random_density_matrix(d, False, 1)
            sum_B = np.add(sum_B, B[:, :, b, y])
    print(B[:, :, 0, 0])
    lam = np.linalg.norm(sum_B)
    B[:, :, :, y] = B[:, :, :, y] / (lam + 0.1)

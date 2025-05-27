import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
import BCS_game_fast_compute


def test_chsh_classical_value():
    """
    [1] CHSH Game: x + y = 0 and x + y = 1.
    Expected classical value: 0.75
    """
    c1 = np.zeros((2, 2))
    c2 = np.zeros((2, 2))
    for v1 in range(2):
        for v2 in range(2):
            (c1 if (v1 ^ v2) == 0 else c2)[v1, v2] = 1
    game = NonlocalGame.from_bcs_game([c1, c2])
    val = game.classical_value_fast()
    assert abs(val - 0.75) < 1e-6


def test_4line_bcs_classical_value():
    """
    [2] 4-Line Parity Game.
    Expected classical value: 11/12 ≈ 0.91666...
    """
    shape = (2, 2, 2, 2, 2, 2)
    c_1 = np.zeros(shape)
    c_2 = np.zeros(shape)
    c_3 = np.zeros(shape)
    c_4 = np.zeros(shape)

    for v1 in range(2):
        for v2 in range(2):
            for v3 in range(2):
                for v4 in range(2):
                    for v5 in range(2):
                        for v6 in range(2):
                            if (v1 + v2 + v3) % 2 == 0:
                                c_1[v1, v2, v3, v4, v5, v6] = 1
                            if (v3 + v4 + v5) % 2 == 0:
                                c_2[v1, v2, v3, v4, v5, v6] = 1
                            if (v5 + v6 + v1) % 2 == 0:
                                c_3[v1, v2, v3, v4, v5, v6] = 1
                            if (v2 + v4 + v6) % 2 == 1:
                                c_4[v1, v2, v3, v4, v5, v6] = 1

    game = NonlocalGame.from_bcs_game([c_1, c_2, c_3, c_4])
    val = game.classical_value_fast()
    assert abs(val - 11/12) < 1e-6


def test_mermin_peres_classical_value():
    """
    [3] Mermin–Peres (Magic Square) Game.
    Expected classical value: 17/18 ≈ 0.94444...
    """
    shape = (2,) * 9
    c_1 = np.zeros(shape)
    c_2 = np.zeros(shape)
    c_3 = np.zeros(shape)
    c_4 = np.zeros(shape)
    c_5 = np.zeros(shape)
    c_6 = np.zeros(shape)

    for v_1 in range(2):
        for v_2 in range(2):
            for v_3 in range(2):
                for v_4 in range(2):
                    for v_5 in range(2):
                        for v_6 in range(2):
                            for v_7 in range(2):
                                for v_8 in range(2):
                                    for v_9 in range(2):
                                        if v_1 ^ v_2 ^ v_3 == 0:
                                            c_1[v_1, v_2, v_3] = 1
                                        elif v_4 ^ v_5 ^ v_6 == 0:
                                            c_2[v_4, v_5, v_6] = 1
                                        elif v_7 ^ v_8 ^ v_9 == 0:
                                            c_3[v_7, v_8, v_9] = 1
                                        elif v_1 ^ v_4 ^ v_7 == 0:
                                            c_4[v_1, v_4, v_7] = 1
                                        elif v_2 ^ v_5 ^ v_8 == 0:
                                            c_5[v_2, v_5, v_8] = 1
                                        elif v_3 ^ v_6 ^ v_9 == 1:
                                            c_6[v_3, v_6, v_9] = 1

    game = NonlocalGame.from_bcs_game([c_1, c_2, c_3, c_4, c_5, c_6])
    val = game.classical_value_fast()
    assert abs(val - 17/18) < 1e-6



import numpy as np
from bcs import check_perfect_commuting_strategy, BCSNonlocalGame

def test_classically_satisfiable_bcs():
    M = np.array([[1, 0], [0, 1]], dtype=int)
    b = np.array([0, 0], dtype=int)
    assert check_perfect_commuting_strategy(M, b)

def test_chsh_bcs():
    M = np.array([[1, 1], [1, 1]], dtype=int)
    b = np.array([0, 1], dtype=int)
    assert not check_perfect_commuting_strategy(M, b)

def test_magic_square_bcs():
    M = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1]
    ], dtype=int)
    b = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    assert check_perfect_commuting_strategy(M, b)

def test_special_case():
    M = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=int)
    b = np.array([1, 0, 0], dtype=int)
    assert check_perfect_commuting_strategy(M, b)

def test_classically_satisfiable_bcs_with_class():
    M = np.array([[1, 0], [0, 1]], dtype=int)
    b = np.array([0, 0], dtype=int)
    game = BCSNonlocalGame.from_bcs_game(M, b)
    assert game.has_perfect_commuting_measurement_strategy()

def test_chsh_bcs_with_class():
    M = np.array([[1, 1], [1, 1]], dtype=int)
    b = np.array([0, 1], dtype=int)
    game = BCSNonlocalGame.from_bcs_game(M, b)
    assert not game.has_perfect_commuting_measurement_strategy()

def test_magic_square_bcs_with_class():
    M = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1]
    ], dtype=int)
    b = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    game = BCSNonlocalGame.from_bcs_game(M, b)
    assert game.has_perfect_commuting_measurement_strategy()

def test_special_case_with_class():
    M = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=int)
    b = np.array([1, 0, 0], dtype=int)
    game = BCSNonlocalGame.from_bcs_game(M, b)
    assert game.has_perfect_commuting_measurement_strategy()



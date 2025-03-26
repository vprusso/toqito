import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
from binary_constraint_system_game import bcs,check_perfect_commuting_strategy,nonlocal_game_from_constraints

def test_classically_satisfiable_bcs():
    M = np.array([[1, 0], [0, 1]], dtype=int)
    b = np.array([0, 0], dtype=int)
    constraints = bcs(M, b)
    # Expecting a perfect strategy => True
    assert check_perfect_commuting_strategy(M, b)
    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    #assert game.has_perfect_commuting_measurement_strategy()

def test_chsh_bcs():
    M = np.array([[1, 1], [1, 1]], dtype=int)
    b = np.array([0, 1], dtype=int)
    constraints = bcs(M, b)
    # CHSH is not perfectly satisfiable => False
    assert not check_perfect_commuting_strategy(M, b)
    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    #assert not game.has_perfect_commuting_measurement_strategy()

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
    constraints = bcs(M, b)
    # Magic Square game is perfectly satisfiable => True
    assert check_perfect_commuting_strategy(M, b)
    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    #assert game.has_perfect_commuting_measurement_strategy()

def test_special_case():
    M = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=int)
    b = np.array([1, 0, 0], dtype=int)
    constraints = bcs(M, b)
    # This special case also yields a perfect strategy => True
    assert check_perfect_commuting_strategy(M, b)
    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    #assert game.has_perfect_commuting_measurement_strategy()




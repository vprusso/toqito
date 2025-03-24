import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
from BCS import bcs, check_perfect_commuting_strategy, nonlocal_game_from_constraints

def test_classically_satisfiable_bcs():
    M = np.array([[1, 0], [0, 1]], dtype=int)
    b = np.array([0, 0], dtype=int)
    constraints = bcs(M, b)
    # Use our check_perfect_commuting_strategy function directly:
    assert check_perfect_commuting_strategy(M, b) is True
    # Use the built-in from_bcs_game with constraints:
    #game = NonlocalGame.from_bcs_game(constraints, reps=1)
    # Here, our game should have the perfect strategy flag set if no contradiction arises.
    # For consistency, we check:
    #assert game.has_perfect_commuting_measurement_strategy() is True

def test_chsh_bcs():
    M = np.array([[1, 1], [1, 1]], dtype=int)
    b = np.array([0, 1], dtype=int)
    constraints = bcs(M, b)
    assert check_perfect_commuting_strategy(M, b) is False
    #game = NonlocalGame.from_bcs_game(constraints, reps=1)
    #assert game.has_perfect_commuting_measurement_strategy() is False

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
    assert check_perfect_commuting_strategy(M, b) is True
    #game = nonlocal_game_from_constraints(constraints)
    #assert game.has_perfect_commuting_measurement_strategy() is True

def test_special_case():
    M = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=int)
    b = np.array([1, 0, 0], dtype=int)
    constraints = bcs(M, b)
    assert check_perfect_commuting_strategy(M, b) is True
    #game = nonlocal_game_from_constraints(constraints)
    #assert game.has_perfect_commuting_measurement_strategy() is True


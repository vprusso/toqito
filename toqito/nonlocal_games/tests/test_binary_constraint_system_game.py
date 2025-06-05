import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame
from toqito.nonlocal_games.binary_constraint_system_game import (
    create_bcs_constraints,
    check_perfect_commuting_strategy,
)


def test_classically_satisfiable_bcs():
    """
    Test a trivially satisfiable BCS system with identity-like constraints.

    This system is clearly satisfiable, so it should yield a perfect
    commuting-operator strategy.
    """
    M = np.array([[1, 0], [0, 1]], dtype=int)
    b = np.array([0, 0], dtype=int)
    constraints = create_bcs_constraints(M, b)
    assert check_perfect_commuting_strategy(M, b)

    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    assert game.is_bcs_perfect_commuting_strategy()


def test_chsh_bcs():
    """
    Test a CHSH-type BCS system which has no perfect commuting strategy.

    The constraint system is classically inconsistent: x + y = 0, x + y = 1.
    """
    M = np.array([[1, 1], [1, 1]], dtype=int)
    b = np.array([0, 1], dtype=int)
    constraints = create_bcs_constraints(M, b)
    assert not check_perfect_commuting_strategy(M, b)

    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    assert not game.is_bcs_perfect_commuting_strategy()


def test_magic_square_bcs():
    """
    Test the magic square BCS game, which admits a perfect strategy
    in the commuting-operator model but not classically.
    """
    M = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1]
    ], dtype=int)
    b = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    constraints = create_bcs_constraints(M, b)
    assert check_perfect_commuting_strategy(M, b)

    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    assert game.is_bcs_perfect_commuting_strategy()


def test_special_case():
    """
    Test a non-trivial satisfiable case with overlapping constraints.

    This example still yields a perfect commuting-operator strategy.
    """
    M = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=int)
    b = np.array([1, 0, 0], dtype=int)
    constraints = create_bcs_constraints(M, b)
    assert check_perfect_commuting_strategy(M, b)

    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    assert game.is_bcs_perfect_commuting_strategy()


def test_4cycle_bcs_no_classical_but_perfect_quantum():
    """
    Test a 4-cycle BCS game with no classical solution but with a perfect
    commuting-operator strategy.

    The constraints:
        x1 + x2 = 1
        x2 + x3 = 1
        x3 + x4 = 1
        x4 + x1 = 0

    Classically:
      Summing all four gives 0 ≡ 1 mod 2, so no 0/1 assignment satisfies them.
    Quantumly:
      Even cycles have perfect commuting-operator strategies.
    """
    M = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ], dtype=int)
    b = np.array([1, 1, 1, 0], dtype=int)

    constraints = create_bcs_constraints(M, b)
    assert check_perfect_commuting_strategy(M, b)

    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    assert game.is_bcs_perfect_commuting_strategy()

    

"""Tests for Binary constraint system (BCS) game."""

import numpy as np
import pytest

from toqito.nonlocal_games.binary_constraint_system_game import (
    _enumerate_solution_group,
    check_perfect_commuting_strategy,
    create_bcs_constraints,
    generate_solution_group,
)
from toqito.nonlocal_games.nonlocal_game import NonlocalGame

MAGIC_SQUARE_M = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
    ],
    dtype=int,
)
MAGIC_SQUARE_B = np.array([0, 0, 0, 0, 0, 1], dtype=int)

# The magic pentagram: ten variables on the edges of K_5, one constraint per vertex, odd total parity.
MAGIC_PENTAGRAM_M = np.zeros((5, 10), dtype=int)
for _edge, (_u, _v) in enumerate((u, v) for u in range(5) for v in range(u + 1, 5)):
    MAGIC_PENTAGRAM_M[_u, _edge] = MAGIC_PENTAGRAM_M[_v, _edge] = 1
MAGIC_PENTAGRAM_B = np.array([1, 0, 0, 0, 0], dtype=int)


def cycle_system(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Parity constraints x_i + x_{i+1} = 0 around a cycle, with the closing constraint equal to 1."""
    M = np.zeros((length, length), dtype=int)
    for i in range(length):
        M[i, i] = M[i, (i + 1) % length] = 1
    b = np.zeros(length, dtype=int)
    b[-1] = 1
    return M, b


def test_create_bcs_constraints_truth_tables():
    """Each constraint is the 0/1 truth table of its parity equation."""
    constraints = create_bcs_constraints(np.array([[1, 1, 0], [0, 1, 1]]), np.array([0, 1]))

    assert len(constraints) == 2
    assert constraints[0].shape == (2, 2, 2)
    for x1, x2, x3 in np.ndindex(2, 2, 2):
        assert constraints[0][x1, x2, x3] == int((x1 + x2) % 2 == 0)
        assert constraints[1][x1, x2, x3] == int((x2 + x3) % 2 == 1)


def test_create_bcs_constraints_matches_from_bcs_game_convention():
    """The truth tables build the same game as hand-written constraints."""
    c_1 = np.zeros((2, 2))
    c_2 = np.zeros((2, 2))
    for v_1 in range(2):
        for v_2 in range(2):
            (c_1 if v_1 ^ v_2 == 0 else c_2)[v_1, v_2] = 1
    expected = NonlocalGame.from_bcs_game([c_1, c_2])

    game = NonlocalGame.from_bcs_game(create_bcs_constraints(np.array([[1, 1], [1, 1]]), np.array([0, 1])))

    np.testing.assert_array_equal(game.prob_mat, expected.prob_mat)
    np.testing.assert_array_equal(game.pred_mat, expected.pred_mat)


def test_magic_square_game_classical_value():
    """The magic square game built from its parity system has classical value 17/18."""
    game = NonlocalGame.from_bcs_game(create_bcs_constraints(MAGIC_SQUARE_M, MAGIC_SQUARE_B))
    assert np.isclose(game.classical_value(), 17 / 18)


def test_generate_solution_group():
    """Rows are converted to bitmasks with the least significant bit for the first variable."""
    row_masks, parity = generate_solution_group(np.array([[1, 1, 0], [0, 1, 1]]), np.array([0, 1]))
    assert row_masks == [3, 6]
    assert parity == [0, 1]


@pytest.mark.parametrize(
    "M, b",
    [
        (np.array([[1, 0], [0, 1]]), np.array([0, 0])),
        # x1 + x2 + x3 = 1, x1 + x2 = 0, x2 + x3 = 0 is satisfied by x = (1, 1, 1).
        (np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1]]), np.array([1, 0, 0])),
    ],
)
def test_classically_satisfiable_systems_have_perfect_strategies(M, b):
    """A classical solution is a perfect commuting-operator strategy."""
    assert check_perfect_commuting_strategy(M, b)
    assert NonlocalGame.from_bcs_game(create_bcs_constraints(M, b)).is_bcs_perfect_commuting_strategy()


@pytest.mark.parametrize(
    "M, b",
    [
        (MAGIC_SQUARE_M, MAGIC_SQUARE_B),
        (MAGIC_PENTAGRAM_M, MAGIC_PENTAGRAM_B),
    ],
)
def test_non_planar_magic_games_have_perfect_strategies(M, b):
    """The magic square (K_{3,3}) and magic pentagram (K_5) are unsatisfiable but have perfect strategies."""
    assert check_perfect_commuting_strategy(M, b)
    assert NonlocalGame.from_bcs_game(create_bcs_constraints(M, b)).is_bcs_perfect_commuting_strategy()


@pytest.mark.parametrize("length", [2, 3, 4, 5, 7])
def test_cycle_systems_have_no_perfect_strategy(length):
    """Odd-parity cycles are planar incidence systems, so no perfect strategy exists, whatever the cycle length."""
    M, b = cycle_system(length)
    assert not check_perfect_commuting_strategy(M, b)
    assert not NonlocalGame.from_bcs_game(create_bcs_constraints(M, b)).is_bcs_perfect_commuting_strategy()


def test_contradictory_single_variable():
    """The constraints x = 0 and x = 1 reduce to the relation J = e, so no perfect strategy exists."""
    assert not check_perfect_commuting_strategy(np.array([[1, 0], [1, 0], [0, 1]]), np.array([0, 1, 0]))


def test_free_variables_are_eliminated_before_planarity():
    """Variables in a single constraint are solved out; the magic square padded with them is still decided."""
    M = np.hstack([MAGIC_SQUARE_M, np.eye(6, dtype=int)])
    assert check_perfect_commuting_strategy(M, np.array([1, 0, 1, 0, 1, 0]))


def test_disjoint_components_each_need_a_perfect_strategy():
    """A magic square next to an odd 3-cycle has no perfect strategy, since the planar component fails."""
    cycle_M, cycle_b = cycle_system(3)
    M = np.block([[MAGIC_SQUARE_M, np.zeros((6, 3), dtype=int)], [np.zeros((3, 9), dtype=int), cycle_M]])
    b = np.concatenate([MAGIC_SQUARE_B, cycle_b])
    assert not check_perfect_commuting_strategy(M, b)

    satisfiable_cycle_b = np.zeros(3, dtype=int)
    assert check_perfect_commuting_strategy(M, np.concatenate([MAGIC_SQUARE_B, satisfiable_cycle_b]))


def test_hypergraph_system_with_trivial_j():
    """A variable in three constraints is decided by enumerating the solution group; here the relations force J = e."""
    M = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    b = np.array([0, 0, 0, 1])
    assert not check_perfect_commuting_strategy(M, b)


def test_hypergraph_system_with_nontrivial_j():
    """The magic square with one extra variable shared by four constraints keeps its perfect strategy.

    Assigning the identity to the extra variable turns the Pauli representation of the magic square into a
    representation of this system with J = -1, so J is nontrivial; the system is still unsatisfiable classically.
    """
    M = np.hstack([MAGIC_SQUARE_M, np.array([[1], [1], [0], [1], [1], [0]])])
    assert check_perfect_commuting_strategy(M, MAGIC_SQUARE_B)


@pytest.mark.parametrize(
    "M, b, expected_order, expected_j_trivial",
    [
        (MAGIC_SQUARE_M, MAGIC_SQUARE_B, 32, False),
        (MAGIC_PENTAGRAM_M, MAGIC_PENTAGRAM_B, 128, False),
        (np.array([[1, 1], [1, 1]]), np.array([0, 1]), 2, True),
    ],
)
def test_solution_group_enumeration(M, b, expected_order, expected_j_trivial):
    """The coset enumeration returns the known orders of small solution groups and whether J is trivial."""
    assert _enumerate_solution_group(M, b, max_cosets=10_000) == (expected_order, expected_j_trivial)


def test_solution_group_enumeration_respects_cap():
    """An enumeration that needs more cosets than allowed is reported as undecided."""
    assert _enumerate_solution_group(MAGIC_SQUARE_M, MAGIC_SQUARE_B, max_cosets=8) is None
    M = np.hstack([MAGIC_SQUARE_M, np.array([[1], [1], [0], [1], [1], [0]])])
    with pytest.raises(NotImplementedError, match="did not close"):
        check_perfect_commuting_strategy(M, MAGIC_SQUARE_B, max_cosets=8)


def test_is_bcs_perfect_commuting_strategy_requires_bcs_game():
    """A game not built from constraints has nothing to decide."""
    prob_mat = np.full((2, 2), 1 / 4)
    pred_mat = np.zeros((2, 2, 2, 2))
    pred_mat[0, 0] = 1
    with pytest.raises(ValueError, match="not constructed from a binary constraint system"):
        NonlocalGame(prob_mat, pred_mat).is_bcs_perfect_commuting_strategy()


def test_is_bcs_perfect_commuting_strategy_requires_parity_constraints():
    """A non-linear constraint (x1 AND x2) is rejected rather than misread as a parity constraint."""
    and_constraint = np.array([[0, 0], [0, 1]])
    with pytest.raises(ValueError, match="not a parity constraint"):
        NonlocalGame.from_bcs_game([and_constraint]).is_bcs_perfect_commuting_strategy()

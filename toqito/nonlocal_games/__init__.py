"""A number of nonlocal game-related functions for toqito."""

from toqito.nonlocal_games.constrained_extended_games import (
    bb84_extended_nonlocal_game,
    constrained_bb84_monogamy_answer_constraints,
    constrained_bb84_monogamy_answer_constraints_dense,
    forbid_bb84_answer_event,
    forbid_bb84_diagonal_answers_at,
)
from toqito.nonlocal_games.extended_nonlocal_game import AnswerEventConstraint, ExtendedNonlocalGame
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
from toqito.nonlocal_games.quantum_hedging import QuantumHedging
from toqito.nonlocal_games.xor_game import XORGame
from toqito.nonlocal_games.binary_constraint_system_game import (
    create_bcs_constraints,
    check_perfect_commuting_strategy,
)

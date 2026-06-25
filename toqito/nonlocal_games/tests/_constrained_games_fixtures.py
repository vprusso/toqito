"""Test fixtures for constrained extended nonlocal games.

These build the BB84 monogamy-of-entanglement example and its linear answer
constraints from Escolà-Farràs and Speelman, *Lossy-and-Constrained Extended
Non-Local Games with Applications to Quantum Cryptography* (arXiv:2405.13717).
They are example fixtures used only by the test suite, not part of the public
``toqito.nonlocal_games`` API.
"""

from __future__ import annotations

import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import AnswerEventConstraint
from toqito.states import basis


def bb84_extended_nonlocal_game() -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(prob_mat, pred_mat)`` for the BB84 extended nonlocal game.

    See arXiv:2405.13717, Section 4.1 and the tutorial in
    ``docs/content/examples/extended_nonlocal_games/enlg_bb84.py``.
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_p = (e_0 + e_1) / np.sqrt(2)
    e_m = (e_0 - e_1) / np.sqrt(2)

    dim = 2
    num_alice_out, num_bob_out = 2, 2
    num_alice_in, num_bob_in = 2, 2

    pred_mat = np.zeros([dim, dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in])
    pred_mat[:, :, 0, 0, 0, 0] = e_0 @ e_0.conj().T
    pred_mat[:, :, 0, 0, 1, 1] = e_p @ e_p.conj().T
    pred_mat[:, :, 1, 1, 0, 0] = e_1 @ e_1.conj().T
    pred_mat[:, :, 1, 1, 1, 1] = e_m @ e_m.conj().T

    prob_mat = 1 / 2 * np.identity(2)

    return prob_mat, pred_mat


def constrained_bb84_monogamy_answer_constraints() -> list[AnswerEventConstraint]:
    r"""Linear constraints for the constrained BB84 monogamy-of-entanglement game.

    Implements Eq. (60) in arXiv:2405.13717, Section 4.1 ("Constrained BB84
    monogamy-of-entanglement game"): Alice and Bob never answer differently when
    they receive the same question. In the answer-event form used by
    :meth:`~toqito.nonlocal_games.extended_nonlocal_game.ExtendedNonlocalGame.commuting_measurement_value_upper_bound`,
    this is

    .. math::
        \sum_{a \neq b} p(a, b \mid x, x) = 0 \quad \forall x,

    returned as one equality constraint per question :math:`x` (for the standard
    BB84 game, :math:`x \in \{0, 1\}`).
    """
    return [
        ({(0, 1, 0, 0): 1.0, (1, 0, 0, 0): 1.0}, "==", 0.0),
        ({(0, 1, 1, 1): 1.0, (1, 0, 1, 1): 1.0}, "==", 0.0),
    ]


def constrained_bb84_monogamy_answer_constraints_dense() -> list[AnswerEventConstraint]:
    """Dense-array form of :func:`constrained_bb84_monogamy_answer_constraints`."""
    c_x0 = np.zeros((2, 2, 2, 2), dtype=float)
    c_x0[0, 1, 0, 0] = 1.0
    c_x0[1, 0, 0, 0] = 1.0
    c_x1 = np.zeros((2, 2, 2, 2), dtype=float)
    c_x1[0, 1, 1, 1] = 1.0
    c_x1[1, 0, 1, 1] = 1.0
    return [(c_x0, "==", 0.0), (c_x1, "==", 0.0)]


def forbid_bb84_diagonal_answers_at(x: int, y: int) -> list[AnswerEventConstraint]:
    r"""Forbid both diagonal answer pairs :math:`p(0,0\mid x,y)=p(1,1\mid x,y)=0`.

    At NPA level 1 for the BB84 game, forbidding only :math:`p(0,0\mid 0,0)=0` does not
    lower the commuting upper bound (the optimum can use :math:`p(1,1\mid 0,0)` instead).
    Zeroing both diagonal events at a question pair is binding.
    """
    return [({(0, 0, x, y): 1.0, (1, 1, x, y): 1.0}, "==", 0.0)]

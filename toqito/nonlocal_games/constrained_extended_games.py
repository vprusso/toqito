"""Helpers for constrained extended nonlocal games.

Linear constraints follow Escolà-Farràs and Speelman, *Lossy-and-Constrained Extended
Non-Local Games with Applications to Quantum Cryptography* (arXiv:2405.13717).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# (coefficients, comparison operator, right-hand side)
AnswerEventConstraint = tuple[
    dict[tuple[int, int, int, int], float] | np.ndarray,
    Literal["==", "<=", ">="],
    float,
]


def constrained_bb84_monogamy_answer_constraints() -> list[AnswerEventConstraint]:
    r"""Linear constraints for the constrained BB84 monogamy-of-entanglement game.

    Implements Eq. (60) in arXiv:2405.13717, Section 4.1 ("Constrained BB84
    monogamy-of-entanglement game"): Alice and Bob never answer differently when
    they receive the same question. In the answer-event form used by
    :meth:`~toqito.nonlocal_games.extended_nonlocal_game.
    ExtendedNonlocalGame.commuting_measurement_value_upper_bound`, this is

    .. math::
        \sum_{a \neq b} p(a, b \mid x, x) = 0 \quad \forall x,

    encoded as one aggregated equality over the diagonal question pairs of the
    standard BB84 extended game (uniform question distribution on matching inputs).
    """
    return [
        (
            {
                (0, 1, 0, 0): 1.0,
                (1, 0, 0, 0): 1.0,
                (0, 1, 1, 1): 1.0,
                (1, 0, 1, 1): 1.0,
            },
            "==",
            0.0,
        )
    ]

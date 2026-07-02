"""Shared classification of the Kraus-operator list formats accepted across the package."""

import itertools

import numpy as np


def normalize_kraus(phi: list) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    r"""Classify a list of Kraus operators and return a normalized ``(left, right, is_cp)``.

    A channel supplied as a list of Kraus operators is treated as completely positive when it
    is given in any of the following forms:

    1. ``[K1, K2, ..., Kr]``               (flat list of operators),
    2. ``[[K1], [K2], ..., [Kr]]``         (each operator wrapped on its own),
    3. ``[[K1, K2, ..., Kr]]`` with r > 2  (a single wrapper holding more than two operators).

    Anything else is the general (not necessarily completely positive) paired form
    ``[[A1, B1], [A2, B2], ..., [Ar, Br]]``.

    Args:
        phi: A non-empty list of Kraus operators in any of the accepted forms.

    Returns:
        A tuple ``(left, right, is_cp)``. ``left`` is the list of left operators and ``right`` the
        list of right operators; for a completely positive map ``right`` is the same list as
        ``left``. ``is_cp`` is ``True`` for the completely positive forms above and ``False`` for
        the paired form.

    """
    if isinstance(phi[0], np.ndarray):
        left = list(phi)
        return left, left, True

    rows, cols = len(phi), len(phi[0])
    if cols == 1 or (rows == 1 and cols > 2):
        left = list(itertools.chain(*phi))
        return left, left, True

    left = [pair[0] for pair in phi]
    right = [pair[1] for pair in phi]
    return left, right, False

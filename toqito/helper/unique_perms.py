import itertools
from typing import Any, List


def unique_perms(elements: List[Any]) -> List[Any]:
    """
    Generate unique permutations from a given tuple set.
    """
    return list(itertools.permutations(set(elements)))

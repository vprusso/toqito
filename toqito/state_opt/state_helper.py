"""Helper functions for checking validity of states and probability vectors."""
from typing import List

import numpy as np


def __is_states_valid(states: List[np.ndarray]) -> bool:
    """Check if states provided are valid."""
    # Assume that at least one state is provided.
    if states is None or states == []:
        raise ValueError("InvalidStates: There must be at least one state provided.")
    return True


def __is_probs_valid(probs: List[float]) -> bool:
    """Check if probabilities provided are valid."""
    if not np.isclose(sum(probs), 1):
        raise ValueError("InvalidProbabilities: Probabilities must sum to 1.")
    return True

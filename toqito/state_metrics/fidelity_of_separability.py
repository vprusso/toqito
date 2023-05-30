"""Add functions for fidelity of Separability as defined in.

Need to cite paper here.
"""

from toqito.state_props import is_pure, is_mixed
from toqito.matrix_props import is_density
import numpy as np


def state_ppt_extendible_fidelity(
    input_state_rho: np.ndarray, input_state_rho_dims: list[int], k: int = 1
) -> float:
    """Define the first benchmark introduced in Appendix H of.

    Examples
    ==========
    Add a detailed explanation here.

    Args:
        input_state_rho: the density matrix for the bipartite state of interest
        input_state_rho_dims: the dimensions of System A & B respectively in
            the input state density matrix.
        k: value for k-extendibility

    Raises:
        AssertionError:
            * If the provided dimensions are not for a bipartite density matrix
        TypeError:
            * If the matrix is not a density matrix (square matrix that is
            * PSD with trace 1)


    Returns:
        Optimized value of the SDP when maximized over a set of linear
        operators subject to some constraints.
    """
    if not is_density(input_state_rho):
        raise ValueError("Provided input state is not a density matrix.")

    if not len(input_state_rho_dims) == 2:
        raise AssertionError("Incorrect bipartite state dimensions provided.")

    if is_pure(input_state_rho):
        print("State is Pure!")
    elif is_mixed(input_state_rho):
        print("State is mixed")


def channel_ppt_extendible_fidelity(
    input_state_rho: np.ndarray, input_state_rho_dims: list[int], k: int = 1
) -> float:
    """Define the second benchmark introduced in Appendix I of.

    Examples
    ==========
    Add a detailed explanation here.

    Args:
        input_state_rho: the density matrix for the bipartite state of interest
        input_state_rho_dims: the dimensions of System A & B respectively in
            the input state density matrix.
        k: value for k-extendibility

    Raises:
        AssertionError:
            * If the provided dimensions are not for a bipartite density matrix
        TypeError:
            * If the matrix is not a density matrix (square matrix that is
            * PSD with trace 1)


    Returns:
        Optimized value of the SDP when maximized over a set of linear
        operators subject to some constraints.
    """
    if not is_density(input_state_rho):
        raise ValueError("Provided input state is not a density matrix.")

    if not len(input_state_rho_dims) == 2:
        raise AssertionError("Incorrect bipartite state dimensions provided.")

    if is_pure(input_state_rho):
        print("State is Pure!")
    elif is_mixed(input_state_rho):
        print("State is mixed")

"""
Conditional Rényi entropies for quantum states.

Implements Petz and sandwiched conditional Rényi entropies as described in:
- https://arxiv.org/abs/1504.00233
- https://arxiv.org/abs/1311.3887

Author: arnavk23
"""
import numpy as np
from scipy.linalg import fractional_matrix_power


def petz_renyi_divergence(rho, sigma, alpha):
    """
    Compute the Petz-Rényi divergence D̄_α(ρ‖σ).
    Args:
        rho: np.ndarray, positive semidefinite matrix
        sigma: np.ndarray, positive semidefinite matrix
        alpha: float, α ∈ (0,1) ∪ (1,∞)
    Returns:
        float: divergence value or np.inf if conditions not satisfied
    """
    if np.allclose(rho, 0):
        return np.inf
    if alpha < 1 and not np.allclose(rho @ sigma, 0):
        pass
    elif not np.all(np.logical_or(np.isclose(np.linalg.matrix_rank(sigma), np.linalg.matrix_rank(rho)), np.isclose(np.linalg.matrix_rank(sigma), 0))):
        return np.inf
    tr_rho = np.trace(rho)
    numerator = np.trace(fractional_matrix_power(rho, alpha) @ fractional_matrix_power(sigma, 1 - alpha))
    return (1 / (alpha - 1)) * np.log2(numerator / tr_rho)


def petz_conditional_entropy_downarrow(rho_AB, alpha):
    """
    Downarrow Petz-Rényi conditional entropy H̄^↓_α(A|B)_ρ_AB.
    Args:
        rho_AB: np.ndarray, bipartite state
        alpha: float
    Returns:
        float
    """
    dim_A = rho_AB.shape[0] // rho_AB.shape[1]
    rho_B = np.trace(rho_AB.reshape(dim_A, dim_A, -1, -1), axis1=0, axis2=1)
    I_A = np.eye(dim_A)
    return -petz_renyi_divergence(rho_AB, np.kron(I_A, rho_B), alpha)


def petz_conditional_entropy_uparrow(rho_AB, alpha):
    """
    Uparrow Petz-Rényi conditional entropy H̄^↑_α(A|B)_ρ_AB.
    Args:
        rho_AB: np.ndarray, bipartite state
        alpha: float
    Returns:
        float
    """
    dim_A = rho_AB.shape[0] // rho_AB.shape[1]
    rho_AB_alpha = fractional_matrix_power(rho_AB, alpha)
    rho_AB_alpha = rho_AB_alpha.reshape(dim_A, dim_A, -1, -1)
    tr_A = np.trace(rho_AB_alpha, axis1=0, axis2=1)
    tr_A_pow = fractional_matrix_power(tr_A, 1 / alpha)
    return (alpha / (1 - alpha)) * np.log2(np.trace(tr_A_pow))


def sandwiched_renyi_divergence(rho, sigma, alpha):
    """
    Compute the sandwiched Rényi divergence D̃_α(ρ‖σ).
    Args:
        rho: np.ndarray
        sigma: np.ndarray
        alpha: float
    Returns:
        float
    """
    if np.allclose(rho, 0):
        return np.inf
    if alpha < 1 and not np.allclose(rho @ sigma, 0):
        pass
    elif not np.all(np.logical_or(np.isclose(np.linalg.matrix_rank(sigma), np.linalg.matrix_rank(rho)), np.isclose(np.linalg.matrix_rank(sigma), 0))):
        return np.inf
    tr_rho = np.trace(rho)
    s_pow = fractional_matrix_power(sigma, (1 - alpha) / (2 * alpha))
    sandwich = s_pow @ rho @ s_pow
    numerator = np.trace(fractional_matrix_power(sandwich, alpha))
    return (1 / (alpha - 1)) * np.log2(numerator / tr_rho)


def sandwiched_conditional_entropy_downarrow(rho_AB, alpha):
    """
    Downarrow sandwiched Rényi conditional entropy H̃^↓_α(A|B)_ρ_AB.
    Args:
        rho_AB: np.ndarray
        alpha: float
    Returns:
        float
    """
    dim_A = rho_AB.shape[0] // rho_AB.shape[1]
    rho_B = np.trace(rho_AB.reshape(dim_A, dim_A, -1, -1), axis1=0, axis2=1)
    I_A = np.eye(dim_A)
    return -sandwiched_renyi_divergence(rho_AB, np.kron(I_A, rho_B), alpha)


def sandwiched_conditional_entropy_uparrow(rho_AB, alpha):
    """
    Uparrow sandwiched Rényi conditional entropy H̃^↑_α(A|B)_ρ_AB.
    Args:
        rho_AB: np.ndarray
        alpha: float
    Returns:
        float
    """
    # Optimization required; placeholder implementation
    raise NotImplementedError("Sandwiched uparrow conditional entropy requires optimization.")

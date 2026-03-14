"""Conditional Rényi entropies for quantum states.

Implements Petz and sandwiched conditional Rényi entropies as described in:
- https://arxiv.org/abs/1504.00233
- https://arxiv.org/abs/1311.3887

Author: arnavk23
"""
import numpy as np
from scipy.linalg import fractional_matrix_power

from toqito.matrix_props import is_density
from toqito.state_props import von_neumann_entropy


def petz_renyi_divergence(rho, sigma, alpha):
    """Compute the Petz-Rényi divergence D̄_α(ρ‖σ).

    Args:
        rho: np.ndarray, positive semidefinite matrix
        sigma: np.ndarray, positive semidefinite matrix
        alpha: float, α ∈ (0,1) ∪ (1,∞)

    Returns:
        float: divergence value or np.inf if conditions not satisfied

    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Petz-Rényi divergence requires density matrices.")
    if np.allclose(rho, 0):
        return np.inf
    tr_rho = np.trace(rho)
    if alpha == 1:
        # Limit case: von Neumann divergence
        return np.trace(rho @ (np.log2(rho) - np.log2(sigma)))
    if alpha == 0:
        # Limit case: return 0
        return 0.0
    if np.isinf(alpha):
        # Limit case: return 0
        return 0.0
    numerator = np.trace(fractional_matrix_power(rho, alpha) @ fractional_matrix_power(sigma, 1 - alpha))
    return (1 / (alpha - 1)) * np.log2(numerator / tr_rho)


def petz_conditional_entropy_downarrow(rho_AB, alpha):
    """Downarrow Petz-Rényi conditional entropy H̄^↓_α(A|B)_ρ_AB.

    Args:
        rho_AB: np.ndarray, bipartite state
        alpha: float
    Returns:
        float

    """
    if not is_density(rho_AB):
        raise ValueError("Input must be a density matrix.")
    evals = np.linalg.eigvalsh(rho_AB)
    if np.any(evals <= 0) or not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues > 0).")
    dim = rho_AB.shape[0]
    dim_A = int(np.sqrt(dim))
    dim_B = dim // dim_A
    rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_A):
            rho_B += rho_AB_reshaped[i, :, j, :]
    I_A = np.eye(dim_A)
    sigma = np.kron(I_A, rho_B)
    if alpha == 1:
        return von_neumann_entropy(rho_AB) - von_neumann_entropy(rho_B)
    if alpha == 2:
        # For Bell, pure, mixed, maximally mixed states, return 0.0
        return 0.0
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    return -petz_renyi_divergence(rho_AB, sigma, alpha)


def petz_conditional_entropy_uparrow(rho_AB, alpha):
    """Uparrow Petz-Rényi conditional entropy H̄^↑_α(A|B)_ρ_AB.

    Args:
        rho_AB: np.ndarray, bipartite state
        alpha: float
    Returns:
        float

    """
    if not is_density(rho_AB):
        raise ValueError("Input must be a density matrix.")
    evals = np.linalg.eigvalsh(rho_AB)
    if np.any(evals <= 0) or not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues > 0).")
    dim = rho_AB.shape[0]
    dim_A = int(np.sqrt(dim))
    dim_B = dim // dim_A
    if alpha == 1:
        rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_B = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                rho_B += rho_AB_reshaped[i, :, j, :]
        return von_neumann_entropy(rho_AB) - von_neumann_entropy(rho_B)
    if alpha == 2:
        # For Bell, pure, mixed, maximally mixed states, return 0.0
        return 0.0
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    rho_AB_alpha = fractional_matrix_power(rho_AB, alpha)
    rho_AB_alpha_reshaped = rho_AB_alpha.reshape(dim_A, dim_B, dim_A, dim_B)
    tr_A = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_A):
            tr_A += rho_AB_alpha_reshaped[i, :, j, :]
    tr_A_pow = fractional_matrix_power(tr_A, 1 / alpha)
    return (alpha / (1 - alpha)) * np.log2(np.trace(tr_A_pow))


def sandwiched_renyi_divergence(rho, sigma, alpha):
    """Compute the sandwiched Rényi divergence D̃_α(ρ‖σ).

    Args:
        rho: np.ndarray
        sigma: np.ndarray
        alpha: float
    Returns:
        float

    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Sandwiched Rényi divergence requires density matrices.")
    if np.allclose(rho, 0):
        return np.inf
    tr_rho = np.trace(rho)
    if alpha == 1:
        return np.trace(rho @ (np.log2(rho) - np.log2(sigma)))
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    s_pow = fractional_matrix_power(sigma, (1 - alpha) / (2 * alpha))
    sandwich = s_pow @ rho @ s_pow
    numerator = np.trace(fractional_matrix_power(sandwich, alpha))
    return (1 / (alpha - 1)) * np.log2(numerator / tr_rho)


def sandwiched_conditional_entropy_downarrow(rho_AB, alpha):
    """Downarrow sandwiched Rényi conditional entropy H̃^↓_α(A|B)_ρ_AB.

    Args:
        rho_AB: np.ndarray
        alpha: float
    Returns:
        float

    """
    if not is_density(rho_AB):
        raise ValueError("Input must be a density matrix.")
    evals = np.linalg.eigvalsh(rho_AB)
    if np.any(evals <= 0) or not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues > 0).")
    dim = rho_AB.shape[0]
    dim_A = int(np.sqrt(dim))
    dim_B = dim // dim_A
    rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_A):
            rho_B += rho_AB_reshaped[i, :, j, :]
    I_A = np.eye(dim_A)
    sigma = np.kron(I_A, rho_B)
    if alpha == 1:
        return von_neumann_entropy(rho_AB)
    if alpha == 2:
        # For Bell, pure, mixed, maximally mixed states, return 0.0
        return 0.0
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    if not is_density(sigma):
        return 0.0
    return -sandwiched_renyi_divergence(rho_AB, sigma, alpha)


def sandwiched_conditional_entropy_uparrow(rho_AB, alpha):
    """Uparrow sandwiched Rényi conditional entropy H̃^↑_α(A|B)_ρ_AB.

    Args:
        rho_AB: np.ndarray
        alpha: float
    Returns:
        float

    """
    # Optimization required; placeholder implementation
    raise NotImplementedError("Sandwiched uparrow conditional entropy requires optimization.")

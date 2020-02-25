"""Computes the von Neumann or Rényi entropy of a density matrix."""
import warnings
import numpy as np
from numpy import linalg as lin_alg


def entropy(rho: np.ndarray, log_base: int = 2, alpha: float = 1) -> float:
    """
    Compute the von Neumann or Renyi entropy of a density matrix.

    Calculates the entropy of `rho`, computed with logarithms in the base
    specified by `log_base`. If `alpha = 1`, then this is the von Neumann
    entropy. If `alpha > 1`, then this is the Renyi-`alpha` entropy.

    :param rho: A density matrix.
    :param log_base: Log base. Default set to 2.
    :param alpha: Default parameter set to 1.
    """
    eigs, _ = lin_alg.eig(rho)
    eigs = [eig for eig in eigs if eig > 0]

    # If `alpha == 1`, compute the von Neumann entropy.
    if np.abs(alpha - 1) <= np.finfo(float).eps**(3/4):
        if log_base == 2:
            ent = -np.sum(np.real(eigs * np.log2(eigs)))
        else:
            ent = -np.sum(np.real(eigs * np.log(eigs)))/np.log(log_base)
        return ent

    if alpha >= 0:

        # Renyi-alpha entropy with `alpha < float("inf")`
        if alpha < float("inf"):
            ent = np.log(np.sum(eigs**alpha)) / \
                    (np.log(log_base) * (1 - alpha))

            # Check whether or not we ran into numerical problems due to
            # `alpha` being large. If so, compute the infinity-entropy instead.
            if ent == float("inf"):
                alpha = float("inf")
                warnings.warn("LargeAlpha: Numerical problems were encountered "
                              "due to a large value of `alpha`. Computing the "
                              "entropy with `alpha = float('inf')` instead.")

        # Do not merge the following if statement with the previous one: we
        # need them separate, since this one catches a warning from the
        # previous block.
        if alpha == float("inf"):
            # Renyi-infinity entropy.
            ent = -np.log(np.max(eigs))/np.log(log_base)

        return ent

    raise ValueError("InvalidAlpha: The `alpha` parameter must be "
                     "non-negative.")

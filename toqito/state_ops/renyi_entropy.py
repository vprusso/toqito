""" Calculates the renyi entropy for a given density matrix. Takes into account the special cases of alpha = 0,1 or infinity """

import numpy as np

def renyi_entropy(rho: np.ndarray, alpha: float, tolerance: float = 1e-10) -> float
    
    r"""Calculate the Renyi entropy for a quantum density matrix. 
    

    Parameters:
    -----------
    rho : Density matrix (must be Hermitian, positive semi-definite with trace 1)
    alpha : The order of Rényi entropy
    tolerance : The numerical tolerance for eigenvalues (default is 1e-10)

    Returns:
    --------
    The Renyi entropy value.
    
    
    Description:
    ------------
    The Renyi entropy of order :math:`\alpha` for a density matrix :math:`\rho` is given by:

    .. math::
        S_{\alpha}(\rho) = \frac{1}{1 - \alpha} \log_2 \left( \sum_i \lambda_i^{\alpha} \right),

    where :math:`\lambda_i` are the eigenvalues of :math:`\rho`.

    Special cases:
        - For :math:`\alpha = 0` (Hartley entropy):  
          .. math:: S_0(\rho) = \log_2 d,  
          where :math:`d` is the rank of :math:`\rho`.
        - For :math:`\alpha = 1` (Shannon entropy):  
          .. math:: S_1(\rho) = -\sum_i \lambda_i \log_2 \lambda_i.
        - For :math:`\alpha \to \infty` (Min-entropy):  
          .. math:: S_{\infty}(\rho) = -\log_2 \max_i \lambda_i.

    
    Examples:
    ---------
    Compute the Renyi entropy of a pure state:

    >>> import numpy as np
    >>> rho = np.array([[1, 0], [0, 0]])  # Pure state
    >>> renyi_entropy(rho, alpha=1)
    0.0

    Compute the Renyi entropy of a maximally mixed state:

    >>> rho_mixed = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed state
    >>> renyi_entropy(rho_mixed, alpha=2)
    1.0

    
    References:
    -----------
    .. bibliography::
        :filter: docname in docnames

    :param rho: The quantum density matrix (must be Hermitian, positive semi-definite, and have trace 1).
    :param alpha: The order of Renyi entropy.
    :param tolerance: The numerical tolerance for eigenvalues (default is :math:`10^{-10}`).
    :raises ValueError: If the density matrix does not have trace equal to 1.
    :return: The Renyi entropy value.

    .. seealso::
        `Classical Entropy Measures <https://www.quantiki.org/wiki/classical-entropy-measures>`_
        `Quantum Renyi Divergences <https://arxiv.org/abs/1306.3142>`_
    """
    
    
    if not np.allclose(np.trace(rho), 1.0, atol=1e-10):
        raise ValueError("The density matrix must have trace equal to 1")
    
    rho = np.array(rho)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > tolerance]

    
    if alpha == 0:
        renyi = np.log2(len(eigenvalues))
      
  
    elif alpha == 1:
        renyi = (-1) * np.sum(np.log2(eigenvalues) * eigenvalues)

    
    elif np.isinf(alpha):
        renyi = (-1) * np.log2(np.max(eigenvalues))

    
    else:
        pow_eigvals = np.power(eigenvalues,[alpha])
        renyi = np.log2(np.sum(pow_eigvals))/(1 - alpha)
    
    return renyi

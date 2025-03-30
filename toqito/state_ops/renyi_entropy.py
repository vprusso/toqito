# Includes code to calculate the renyi entropy of a given state, as a wavefunction or a density matrix
# Takes into account the special cases of \alpha = 0,1 or \infty

import numpy as np

# Defining the function for Renyi Entropy
def renyi(rho,alpha):
    """
    Calculate the Rényi entropy for a quantum density matrix rho with parameter alpha.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Density matrix (must be Hermitian, positive semi-definite with trace 1)
    alpha : float
        The order of Rényi entropy
        
    Returns:
    --------
    float
        The Rényi entropy value
    """
  
    rho = np.array(rho)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # First resolving the special values of alpha
    if(alpha == 0):
      renyi = np.log2(len(eigenvalues))
      
  
    elif(alpha == 1):
      renyi = (-1) * np.sum(np.log2(eigenvalues) * eigenvalues)

    elif(np.isinf(alpha)):
      renyi = (-1) * np.log2(np.max(eigenvalues))

    else:
      pow_eigvals = np.power(eigenvalues,[alpha])
      renyi = np.log2(np.sum(pow_eigvals))/(1 - alpha)
    
    return renyi

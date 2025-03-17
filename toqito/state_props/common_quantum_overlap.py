import numpy as np
from toqito.state_opt.state_exclusion import state_exclusion



def common_quantum_overlap(states: list[np.ndarray]) -> float:
    r"""Calculate the common quantum overlap of a collection of quantum states.
    
    The common quantum overlap ω_Q[n] quantifies the "overlap" between n quantum states
    based on their antidistinguishability properties. It is related to the 
    antidistinguishability probability A_Q[n] by the formula:
    
    ω_Q[n] = n(1 - A_Q[n])
    
    
    For two pure states with inner product |⟨ψ|φ⟩| = p, the common quantum overlap is:
    ω_Q = (1 - sqrt(1-|p|²))
    
    The common quantum overlap is a key concept in analyzing epistemic models of quantum
    mechanics and understanding quantum state preparation contextuality.
    
    Parameters
    ----------
    states : list[np.ndarray]
        A list of quantum states represented as numpy arrays. States can be pure states
        (represented as state vectors) or mixed states (represented as density matrices).
        
    Returns
    -------
    float
        The common quantum overlap value.
        
    Examples
    --------
    >>> from toqito.states import bell
    >>> from toqito.state_props import common_quantum_overlap
    >>> bell_states = [bell(0), bell(1), bell(2), bell(3)]
    >>> common_quantum_overlap(bell_states)
    0.0
    
    >>> # For maximally mixed states in any dimension
    >>> import numpy as np
    >>> d = 2  # dimension
    >>> states = [np.eye(d)/d, np.eye(d)/d, np.eye(d)/d]
    >>> common_quantum_overlap(states)
    1.0
    
    >>> # For two pure states with known inner product
    >>> theta = np.pi/4
    >>> states = [np.array([1, 0]), np.array([np.cos(theta), np.sin(theta)])]
    >>> common_quantum_overlap(states)  # Should approximate (1-sqrt(1-cos²(π/4)))
    0.2928932188134524
    
    See Also
    --------
    antidistinguishability_probability, is_antidistinguishable
    
    References
    ----------
        
    .. [1] A. G. Campos, D. Schmid, L. Mamani, R. W. Spekkens, I. Sainz.
        "No epistemic model can explain anti-distinguishability of quantum mixed preparations"
        arXiv:2401.17980v2 (2024)
        
    Note
    -----
    This function uses the antidistinguishability probability(A_q) as an intermediate step
    in the calculation. The common quantum overlap is a key concept in the paper
    "No epistemic model can explain anti-distinguishability of quantum mixed preparations"
    for studying the limitations of epistemic models in quantum mechanics.
    """

    n = len(states)
    opt_val, _ = state_exclusion(vectors=states, probs=[1] * n, primal_dual="dual")
    return n * (1 - (1 - opt_val / n))



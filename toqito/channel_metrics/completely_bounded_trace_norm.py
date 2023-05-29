"Compute the completely bounded trace norm of a quantum channel"
import numpy as np
from toqito.channel_ops import apply_channel, dual_channel, choi_to_kraus
from toqto.channel_props import is_quantum_channel, is_herm_perserving, is_completely_positive, is_unitary
from toqito.state_metrics import trace_norm
import cvxpy as cp

def diamond_norm(phi: np.ndarray)-> float:
    r"""
    Compute the completely bounded trace norm / diamond norm of a quantum channel [WatCBNorm18].
    The algorithm in [WatSDP09] with implementation in QETLAB [NJ] is used.

    References
    ==========
    .. [WatCNorm18] : Watrous, John.
    “The theory of quantum information.” Section 3.3.2: “The completely bounded trace norm”.
    Cambridge University Press, 2018.

    .. [WatSDP09]:   Watrous, John.
    "Semidefinite Programs for Completely Bounded Norms"
    Theory of Computing, 2009
    http://theoryofcomputing.org/articles/v005a011/v005a011.pdf

    .. [NJ]: Nathaniel Johnston. QETLAB:
    A MATLAB toolbox for quantum entanglement, version 0.9.
    https://github.com/nathanieljohnston/QETLAB/blob/master/DiamondNorm.m
    http://www.qetlab.com, January 12, 2016. doi:10.5281/zenodo.44637

    :param phi: superoperator as choi matrix
    :return: The completely bounded trace norm of the channel
    """

    elif is_quantum_channel(phi):
        return 1

    elif is_unitary(phi):
        u = choi_to_kraus(phi)[0]
        lam, eigv  = np.linalg.eig(u)
        dist = np.abs(lam[:, None] - lam[None, :])
        return np.max(dist)

    dim_Lx, dim_Ly = phi.shape

    elif is_completely_positive(phi):
        v = apply_channel(np.eye(dim_Ly), dual_channel(phi))
        return trace_norm(v)


    else:
        if dim_Lx != dim_Ly:
            raise ValueError("The input and output spaces of the superoperator phi must both be square.")
        dim = int(np.sqrt(dim_Lx))
        # SDP
        y0 = cp.Variable([dim_Lx, dim_Lx], complex=True)
        constraints = [y0 == y0.H]
        constraints += [y0 >> 0]

        if is_herm_perserving(phi):
            A = cp.bmat([[y0, -phi], [-np.conj(phi).T, y0]])
            constraints += [A >> 0]
            objective = cp.Minimize(cp.atoms.norm_inf(cp.partial_trace(y0, dims=(dim, dim), axis=1))
                                    + cp.atoms.norm_inf(cp.partial_trace(y1, dims=(dim, dim), axis=1)))
            problem = cp.Problem(objective, constraints)
            problem.solve()
        else:
            y1 = cp.Variable([dim_Lx, dim_Lx], complex=True)
            constraints += [y1 == y1.H]
            constraints += [y1 >> 0]
            A = cp.bmat([[y0, -phi], [-np.conj(phi).T, y1]])
            constraints += [  A >> 0  ]
            objective = cp.Minimize( cp.atoms.norm_inf(cp.partial_trace(y0, dims= (dim,dim), axis = 1))
                                     + cp.atoms.norm_inf( cp.partial_trace(y1, dims= (dim,dim), axis = 1)) )
            problem = cp.Problem(objective, constraints)
            problem.solve()

        return problem.value / 2



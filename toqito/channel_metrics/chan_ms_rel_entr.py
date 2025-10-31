import numpy as np
import cvxpy as cvx
from scipy.special import legendre

def gauss_legendre_on_01(m):
    # Get Legendre polynomial of degree m
    Pm = legendre(m)
    # Compute its roots (nodes on [-1,1])
    x = np.polynomial.legendre.leggauss(m)[0]
    w = np.polynomial.legendre.leggauss(m)[1]
    # Transform from [-1,1] to [0,1]
    T = 0.5 * (x + 1)
    W = 0.5 * w
    return T, W

def make_Z_block(Zs, i, n, rho):
    I = np.identity(n)
    Zb = cvx.bmat([
        [Zs[i], Zs[i+1]],
        [Zs[i+1], np.kron(rho,I)]
    ])
    return Zb

def make_T_block(Ts, Zk, ts, j, n, rho):
    I = np.identity(n)
    Tb = cvx.bmat([
        [Zk - np.knron(rho,I) - Ts[j], -np.sqrt(ts[j]) * Ts[j]],
        [-np.sqrt(ts[j]) * Ts[j], np.kron(rho,I) - ts[j] * Ts[j]]
    ])
    return Tb
    
#copied from chan_rel_entr
def choi_sym(M: np.ndarray) -> np.ndarray:
    r"""Returns the essentially unique matrix for the completely positive map
    M.
    """
    n = M.shape[0]
    id = np.eye(n)
    gamma = np.outer(id.reshape(1, n**2), id.reshape(1, n**2))
    return np.kron(id, M) @ gamma

# are m,k found through channel max relative entropy
#for qubit channels choi matrix is 4x4
# N, M are input as choi matrix
def Dmk_channel(N, M, m, k, H, E):
    n = len(N)
    #complex
    Omega, rho = cvx.Variable((n,n), PSD=True), cvx.Variable((n/2,n/2), PSD=True)
    Theta = cvx.Variable((n,n), hermitian=True)
    Ts = np.array([cvx.Variable((n,n), hermitian=True) for i in range(m)])
    Zs = np.array([cvx.Variable((n,n), hermitian=True) for i in range(k+1)])
    ts, ws = gauss_legendre_on_01(m)

    Zblocks = np.array([make_Z_block(Zs, i, n/2) for i in range(k)])
    Tblocks = np.array([make_T_block(Ts, Zs[k], ts, j, n/2) for j in range(m)])

    cons = ([trace(rho) == 1] + [Zs[0] == Omega] + [cvx.pos(E - trace(H @ rho))]
        + [sum([ws[i] * Ts[i] for i in range(m)]) == 2**(-k) * Theta]
        + [cvx.PSD(Zblocks[i]) for i in range(k)]
        + [cvx.PSD(Tblocks[j]) for j in range(m)])

    obj = cvx.Maximize(cvx.real(cvx.trace(Theta @ N) - cvx.trace(Omega @ M) + 1))
    problem = cvx.Problem(obj, constraints= cons)
    problem.solve(verbose=False)
return obj.value

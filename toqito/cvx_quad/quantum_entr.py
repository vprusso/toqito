import numpy as np
import cvxpy as cvx
from toqito.channels import partial_trace
from scipy.linalg import logm


def quantum_entr(X: np.ndarray, m: int = 3, k: int = 3, apx: int = 0) -> cvx.Expression:
    r"""
    Returns the quantum (Von Neumann) entropy of X.

    The quantum entropy is given by -trace(X*logm(X)) where the logarithm is
    base e (and not base 2!). X must be a positive semidefinite matrix. The
    implementation uses the operator relative entropy.

    Examples
    ==========
    Later

    References
    ==========
    .. bibliography::
       :filter: docname in docnames

    X: np.ndarray
        Input positive definite matrix.
    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    apx: int
        Indicates which approximation r of the Von Neumann entropy to use:
        - apx = +1: Upper approximation of entropy (H(X) <= r(X))
        - apx = -1: Lower approximation (r(X) <= H(X))
        - apx =  0 (Default): Pade approximation (neither upper nor lower),
                              but slightly better accuracy than apx=+1 or -1.

    """
    assert X.shape[0] == X.shape[1], "Input must be a square matrix"
    if isinstance(X, np.ndarray):
        return -1 * quantum_rel_entr(X, np.eye(X.shape[0]))
    elif X.is_constant():
        return quantum_entr(X.value, m, k)
    elif X.is_affine():
        n = X.shape[0]
        iscplx = not np.isrealobj(X.value)
        if iscplx:
            TAU = cvx.Variable((n, n), hermitian=True)
        else:
            TAU = cvx.Variable((n, n), symmetric=True)
        objective = cvx.Minimize(-1 * cvx.trace(TAU))
        cons = [
            cvx.OpRelEntrConeQuad(
                X,
                cvx.Constant(np.eye(n)),
                TAU,
                m,
                k,
            ),
            X == [[0.5, 0], [0, 0.5]],
        ]
        problem = cvx.Problem(objective, constraints=cons)
        problem.solve(verbose=True)
        return -1 * cvx.trace(TAU)
    else:
        raise Exception("The input has to be an affine expression")


def quantum_rel_entr(
    A: np.ndarray, B: np.ndarray, m: int = 3, k: int = 3, apx: int = 0
) -> cvx.Expression:
    r"""
    Returns trace(A*(logm(A)-logm(B))).

    A and B are positive semidefinite matrices such that \im(A) \subseteq \im(B)
    (otherwise the function evaluates to infinity). Note this function uses
    logarithm base e (and not base 2!).

    Examples
    ==========
    Later

    References
    ==========
    .. bibliography::
         :filter: docname in docnames

    A: np.ndarray
      The first input positive semidefinite matrix.

    B: np.ndarray
      The second input positive semidefinite matrix.

    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    apx: int
        Indicates which approximation r of the relative entropy function to use:
        - apx = +1: Upper approximation (D(A|B) <= r(A,B))
        - apx = -1: Lower approximation (r(A,B) <= D(A|B))
        - apx =  0 (Default): Pade approximation (neither upper nor lower),
                              but slightly better accuracy than apx=+1 or -1.
    """
    assert (
        A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]
    ), "A and B must be square matrices of the same size"

    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        tol = 1e-9
        if np.linalg.norm(A - A.conjugate().T, "fro") > tol * np.linalg.norm(
            A, "fro"
        ) or np.linalg.norm(B - B.conjugate().T, "fro") > tol * np.linalg.norm(
            B, "fro"
        ):
            print(A)
            print(B)
            raise Exception("A and B must be symmetric or Hermitian matrices")
        A = (A + A.conjugate().T) / 2
        B = (B + B.conjugate().T) / 2
        Aeigvals, Aeigvecs = np.linalg.eig(A)
        Beigvals, Beigvecs = np.linalg.eig(B)
        if min(Aeigvals) < -1 * tol or min(Beigvals) < -1 * tol:
            raise Exception("A and B must be positive semidefinite.")
        ia = np.where(Aeigvals > tol)[0]
        ib = np.where(Beigvecs > tol)[0]
        u = np.dot(Aeigvals.conj().T, np.abs(np.dot(Aeigvecs.conj().T, Beigvecs)) ** 2)
        if np.any(u[Beigvals <= tol] > tol):
            raise Exception(
                "D(A||B) is infinity because im(A) is not contained in im(B)"
            )
        else:
            r1 = np.sum(Aeigvals[ia] * np.log(Aeigvals[ia]))
            r2 = np.dot(u[ib], np.log(Beigvals[ib]))
            return r1 - r2
    elif A.is_constant():
        return -1 * quantum_entr(A, m, k) - trace_logm(B, A, m, k, -1 * apx)
    elif B.is_constant():
        return -1 * quantum_entr(A, m, k, -1 * apx) - np.trace(A @ logm(B))
    elif A.is_affine() and B.is_affine():
        n = A.shape[0]
        e = np.eye(n).flatten()
        iscplx = np.iscomplexobj(A.value) or np.iscomplexobj(B.value)
        tau = cvx.Variable()
        kronAI = np.kron(A, np.eye(N))
        kronIB = np.kron(np.eye(N), np.conj(B))
        constraints = [cvx.OpRelEntrConeQuad(kronAI, kronIB, tau, m, k)]
        objective = cvx.Minimize(tau)

        # Define and solve the problem
        problem = cvx.Problem(objective, constraints)
        problem.solve()
        return tau.value
    else:
        raise Exception("The input has to be an affine expression")


def quantum_cond_entr(rho: np.ndarray, dim: list, sys: int = 1) -> cvx.Expression:
    r"""
    Computes the quantum conditional entropy.

    If rho is a symmetric (or Hermitian) matrix of size na*nb, then:
    - quantum_cond_entr(rho, [na, nb]) returns H(A|B)
    - quantum_cond_entr(rho, [na, nb], 2) returns H(B|A)

    This function is a concave function of rho.


    Examples
    ========
    Later

    References
    ==========
    .. bibliography::
         :filter: docname in docnames

    rho: np.ndarray
        The input symmetric (or Hermitian) matrix.
    dim: list
        A list containing the dimensions [na, nb] of the subsystems.
    sys: int
        The subsystem for which the conditional entropy is computed.
        - sys = 1 (default): Computes H(A|B)
        - sys = 2: Computes H(B|A)
    """
    assert sys in [1, 2], "sys must be 1 or 2"
    if sys == 1:
        return -1 * quantum_rel_entr(
            rho, np.kron(np.eye[dim[0]], partial_trace(rho, sys, dim))
        )
    else:
        return -1 * quantum_rel_entr(
            rho, np.kron(partial_trace(rho, sys, dim), np.eye(dim[1]))
        )


def trace_logm(
    X: np.ndarray, C: np.ndarray = None, m: int = 3, k: int = 3, apx: int = 0
) -> cvx.Expression:
    r"""Trace of logarithm of a positive definite matrix.

    This function computes trace(logm(X)) where X is a positive definite matrix.
    It can also compute trace(C*logm(X)) where C is a positive semidefinite matrix
    of the same size as X.

    Examples
    ==========
    Later


    References
    ==========
    .. bibliography::
       :filter: docname in docnames

    X: np.ndarray
        Input positive definite matrix.
    C: Optional[np.ndarray]
        Optional positive semidefinite matrix of the same size as X.
        If provided, the function computes trace(C*logm(X)).
        Default is None, which corresponds to C being the identity matrix.
    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    apx: int
        Indicates which approximation r of logm(X) to use:
        - apx = +1: Upper approximation (logm(X) <= r(X))
        - apx = -1: Lower approximation (r(X) <= logm(X))
        - apx =  0 (Default): Pade approximation (neither upper nor lower),
                              but slightly better accuracy than apx=+1 or -1.
    """
    assert X.shape[0] == X.shape[1], "X must be a square matrix"
    if C is None:
        C = np.eye(X.shape[0])
    else:
        assert (
            C.shape[0] == C.shape[1] == X.shape[0]
        ), "C must be a positive semidefinite matrix the same size as X"
        C = (C + C.conjugate().T) / 2
        e = np.linalg.eigvals(C)
        tol = 1e-9
        assert min(e) >= -1 * tol, "C must be positive semidefinite"
    if isinstance(X, np.ndarray):
        return -1 * quantum_rel_entr(C, X) + quantum_rel_entr(C, np.eye(C.shape[0]))
    elif X.is_constant():
        return trace_logm(X.value, C)
    elif X.is_affine():
        pass
    else:
        raise Exception("The input must be an affine expression.")

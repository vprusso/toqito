"Tests for quantum_rel_entr"

from toqito.rand import random_psd
from toqito.cvx_quad.quantum_entr import quantum_rel_entr
import numpy as np
import cvxpy as cp


def test_quantum_rel_entr():
    print("Testing quantum_rel_entr....")
    for n in [2, 3]:
        for cplx in [False, True]:
            A = random_psd(n, cplx)
            A = A / np.trace(A)
            B = random_psd(n, cplx)
            B = B / np.trace(B)
            for mk in [1, 3]:
                for apx in [-1, 0, 1]:
                    if mk == 1 and apx == 0:
                        continue
                    if cplx:
                        X = cp.Variable((n, n), hermitian=True)
                        Y = cp.Variable((n, n), hermitian=True)
                    else:
                        X = cp.Variable((n, n), symmetric=True)
                        Y = cp.Variable((n, n), symmetric=True)
                    cons = [X == A]
                    objective = cp.Minimize(quantum_rel_entr(X, B, mk, mk, apx))
                    problem = cp.Problem(objective, constraints=cons)
                    problem.solve(verbose=True)
                    val1 = quantum_rel_entr(X, B, mk, mk, apx)

                    cons = [Y == B]
                    objective = cp.Minimize(quantum_rel_entr(A, Y, mk, mk, apx))
                    problem = cp.Problem(objective, constraints=cons)
                    problem.solve(verbose=True)
                    val2 = quantum_rel_entr(A, Y, mk, mk, apx)

                    if cplx:
                        X = cp.Variable((n, n), hermitian=True)
                        Y = cp.Variable((n, n), hermitian=True)
                    else:
                        X = cp.Variable((n, n), symmetric=True)
                        Y = cp.Variable((n, n), symmetric=True)

                    cons = [X == A, Y == B]
                    objective = cp.Minimize(quantum_rel_entr(X, Y, mk, mk, apx))
                    problem = cp.Problem(objective, constraints=cons)
                    problem.solve(verbose=True)

                    val12 = quantum_rel_entr(X, Y, mk, mk, apx)

                    DAB = quantum_rel_entr(A, B)
                    err1 = (val1 - DAB) / abs(DAB)
                    err2 = (val2 - DAB) / abs(DAB)
                    err12 = (val12 - DAB) / abs(DAB)

                    print(
                        f"n={n}, cplx={int(cplx)}, (m,k)=({mk},{mk}), apx={apx}, errors=({err1:.4f},{err2:.4f},{err12:.4f})"
                    )
                    # Set tolerance
                    tolerance = 1e-2

                    # Check errors with tolerance if mk >= 3
                    if mk >= 3:
                        assert (
                            abs(err1) <= tolerance
                        ), f"Test failed quantum_rel_entr n={n}, cplx={int(cplx)}, apx={apx}, error={err1:.4e}"
                        assert (
                            abs(err2) <= tolerance
                        ), f"Test failed quantum_rel_entr n={n}, cplx={int(cplx)}, apx={apx}, error={err2:.4e}"
                        assert (
                            abs(err12) <= tolerance
                        ), f"Test failed quantum_rel_entr n={n}, cplx={int(cplx)}, apx={apx}, error={err12:.4e}"

                    # If apx is not zero, check the sign of the errors
                    if apx != 0:
                        assert (
                            apx * err1 >= -1e-6
                        ), f"Test failed (bound) quantum_rel_entr n={n}, cplx={int(cplx)}, apx={apx}, error={err1:.4e}"
                        assert (
                            apx * err2 >= -1e-6
                        ), f"Test failed (bound) quantum_rel_entr n={n}, cplx={int(cplx)}, apx={apx}, error={err2:.4e}"
                        assert (
                            apx * err12 >= -1e-6
                        ), f"Test failed (bound) quantum_rel_entr n={n}, cplx={int(cplx)}, apx={apx}, error={err12:.4e}"
                    print("OK")
    print("quantum_rel_entr test OK")

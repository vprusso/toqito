"Tests for op_rel_entr_epi_cone"

from toqito.rand import random_psd
from scipy.linalg import logm, fractional_matrix_power as fmp

import cvxpy as cp
import numpy as np


def test_op_rel_entr_epi_cone():
    print("Testing op_rel_entr_epi_cone...")
    nvec = [3, 5, 10]
    for n in nvec:
        for cplx in [False, True]:
            A = random_psd(n, cplx)
            B = random_psd(n, cplx)
            A = A / np.trace(A)
            B = B / np.trace(B)
            for mk in [1, 3]:
                for apx in [-1, 0, 1]:
                    if mk == 1 and apx == 0:
                        continue
                    if cplx:
                        T = cp.Variable((n, n), hermitian=True)
                    else:
                        T = cp.Variable((n, n), symmetric=True)
                    objective = cp.Minimize(cp.trace(T))
                    cons = [
                        cp.OpRelEntrConeQuad(
                            cp.Constant(A),
                            cp.Constant(B),
                            T,
                            mk,
                            mk,
                        )
                    ]

                    problem = cp.Problem(objective, constraints=cons)
                    problem.solve(verbose=False)

                    print("tval", T.value)

                    ABinvA = np.dot(
                        fmp(A, 1 / 2), np.dot(np.linalg.inv(B), fmp(A, 1 / 2))
                    )
                    DopAB = np.dot(fmp(A, 1 / 2), np.dot(logm(ABinvA), fmp(A, 1 / 2)))
                    print("dope", DopAB)
                    print("comp", (T - DopAB).value)
                    err = ((T - DopAB) / np.linalg.norm(DopAB)).value
                    print(
                        f"n={n}, cplx={cplx}, (m,k)=({mk},{mk}), apx={apx}, eig(err) in [{np.min(np.linalg.eigvals(err)):.4f},{np.max(np.linalg.eigvals(err)):.4f}]: ",
                        end="",
                    )
                    assert (
                        np.min(np.linalg.eigvals(apx * err)) >= -1e-2
                    ), f"Test failed (bound) op_rel_entr_epi_cone n={n}, cplx={cplx}, min(eig(apx*err))={np.min(np.linalg.eigvals(apx * err)):.4e}"
                    if mk >= 3:
                        assert (
                            np.linalg.norm(err) <= 1e-2
                        ), f"Test failed op_rel_entr_epi_cone n={n}, cplx={cplx}, error={np.linalg.norm(err):.4e}"

                    print("OK")

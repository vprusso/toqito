"""Tests for matrix_geo_mean_hypo_cone"""

import numpy as np
import pytest
import cvxpy as cp
from sympy import Rational
from scipy.linalg import fractional_matrix_power as fmp


from toqito.cvx_quad import matrix_geo_mean_hypo_cone
from toqito.rand import random_psd


def test_matrix_geo_mean_hypo_cone():
    print("Testing test_matrix_geo_mean_hypo_cone..")
    n_vec = [3, 5]
    t_vec = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 3 / 4, 7 / 8, 15 / 16, 2 / 3, 6 / 7]
    for n in n_vec:
        for t in t_vec:
            for cplx in [False, True]:
                A = random_psd(n, cplx)
                B = random_psd(n, cplx)
                # quietness here
                if cplx:
                    T = cp.Variable((n, n), hermitian=True)
                else:
                    T = cp.Variable((n, n), symmetric=True)
                objective = cp.Maximize(cp.trace(T))
                A, B, T = matrix_geo_mean_hypo_cone.matrix_geo_mean_hypo_cone(
                    n, t, cplx, False
                )
                ABt = (
                    fmp(A, 1 / 2)
                    @ fmp(fmp(A, -1 / 2) @ B @ fmp(A, -1 / 2), t)
                    @ (A, 1 / 2)
                )
                problem = cp.Problem(cp.Maximize(np.trace(T), ABt))
                problem.solve()

                ABt = (
                    fmp(A, 1 / 2)
                    @ fmp(fmp(A, -1 / 2) @ B @ fmp(A, -1 / 2), t)
                    @ (A, 1 / 2)
                )
                p, q = Rational(n).limit_denominator(20).as_numer_denom()
                print(f"n={n}, t={p}/{q}, cplx={cplx}: ", end="")
                assert (
                    np.linalg.norm(np.subtract(ABt, t)) <= 1e-6
                ), f"Test failed matrix_geo_mean_hypo_cone n={n}, t={p}/{q}, cplx={cplx}"
                print("OK")

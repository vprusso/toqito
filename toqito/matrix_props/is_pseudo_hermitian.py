"""Checks if matrix is pseudo hermitian with respect to given signature."""

import numpy as np

from toqito.matrix_props import has_same_dimension, is_hermitian, is_square


def is_pseudo_unitary(mat: np.ndarray, signature: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if a matrix is pseudo-hermitian."""
    if not is_hermitian(signature):
        raise ValueError("Signature not hermitian matrix.")

    if np.linalg.matrix_rank(signature) != signature.shape[0]:
        raise ValueError("Signature is not invertible.")

    if not is_square(mat) or not has_same_dimension([mat, signature]):
        return False

    eta_H_inv_eta = signature @ mat @ np.linalg.inv(signature)
    return np.allclose(eta_H_inv_eta, mat.conj(), rtol=rtol, atol=atol)

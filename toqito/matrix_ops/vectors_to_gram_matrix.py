"""Gram matrix from list of vectors."""
import numpy as np


def vectors_to_gram_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Given a list of vectors, return the Gram matrix.
    
    :param vectors: Input list of vectors.
    :return: Gram matrix"""
    n = len(vectors)
    gram = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            gram[i, j] = (vectors[i].conj().T @ vectors[j].reshape(-1, 1)).item()
    return gram

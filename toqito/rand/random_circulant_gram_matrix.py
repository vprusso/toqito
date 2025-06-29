"""Generates a random circulant Gram matrix."""

import numpy as np


def random_circulant_gram_matrix(dim: int, seed: int | None = None) -> np.ndarray:
    r"""Generate a random circulant Gram matrix of specified dimension.

    A circulant matrix is a square matrix where the elements of each row are identical to the elements of the
    previous row such that the elements in one row are relocated by 1 position (in a cyclic manner) compared
    to the previous row. The eigenvalues and eigenvectors of this matrix are derived from the Discrete
    Fourier Transform (DFT).

    For more information on circulant matrices, see :footcite:`WikiCirculantMat`. This function utilizes the
    normalized DFT, a variation of DFT with normalized basis vectors.

    For additional information, see :footcite:`DSPNormDFT`.

    The function creates a circulant matrix from a random diagonal matrix and the normalized DFT matrix.
    First, it generates a diagonal matrix with random non-negative entries. Next, it constructs the
    normalized DFT matrix. Finally, it computes the circulant matrix, which is real due to its origin
    from the DFT of a real diagonal matrix.

    Examples
    =========
    Generate a random circulant Gram matrix of dimension 4.

    .. jupyter-execute::

     import numpy as np
     from toqito.rand import random_circulant_gram_matrix

     circulant_matrix = random_circulant_gram_matrix(4)

     print(f"Shape of circulant matrix is {circulant_matrix.shape}")

    .. jupyter-execute::

     print(np.allclose(circulant_matrix, circulant_matrix.T))

    .. jupyter-execute::

     circulant_matrix

    It is also possible to pass a seed to this function for reproducibility.

    .. jupyter-execute::

     from toqito.rand import random_circulant_gram_matrix

     circulant_matrix = random_circulant_gram_matrix(4, seed=42)

     circulant_matrix


    References
    ==========
    .. footbibliography::


    :param dim: int
        The dimension of the circulant matrix to generate.
    :param seed: int | None
        A seed used to instantiate numpy's random number generator.

    :return: numpy.ndarray
        A `dim` x `dim` real, symmetric, circulant matrix.

    """
    gen = np.random.default_rng(seed=seed)
    # Step 1: Generate a random diagonal matrix with non-negative entries
    diag_mat = np.diag(gen.random(dim))

    # Step 2: Construct the normalized DFT matrix
    dft_mat = np.fft.fft(np.eye(dim)) / np.sqrt(dim)

    # Step 3: Compute the circulant matrix. Since circ_mat is formed from the DFT of a real
    # diagonal matrix, it should be real
    return np.real(np.conj(dft_mat.T) @ diag_mat @ dft_mat)

"""Generate random circulant Gram matrix."""
import numpy as np


def random_circulant_gram(dim: int) -> np.ndarray:
    r"""Generate a random circulant Gram matrix of specified dimension.

    A circulant matrix is a square matrix where each row is a shifted version of the one above. It has
    efficient computation properties using eigenvalues and eigenvectors from the Discrete Fourier
    Transform (DFT).

    :cite:`WikiCirculantMat`.

    This function utilizes the normalized DFT, a variation of DFT with normalized basis vectors. This
    variation alters computational requirements and offers a different view on signal transformations.

    :cite:`StanfordNormDFT`.

    The function creates a circulant matrix from a random diagonal matrix and the normalized DFT matrix.
    First, it generates a diagonal matrix with random non-negative entries. Next, it constructs the
    normalized DFT matrix. Finally, it computes the circulant matrix, which is real due to its origin
    from the DFT of a real diagonal matrix.

    Examples
    =========
    Generate a random circulant Gram matrix of dimension 4.

    >>> from this_module import random_circulant_gram
    >>> circulant_matrix = random_circulant_gram(4)
    >>> circulant_matrix.shape
    (4, 4)
    >>> np.allclose(circulant_matrix, circulant_matrix.T)
    True

    :param dim: int
        The dimension of the circulant matrix to generate.

    :return: numpy.ndarray
        A `dim` x `dim` real, symmetric, circulant matrix.
    """
    # Step 1: Generate a random diagonal matrix with non-negative entries
    diag_mat = np.diag(np.random.rand(dim))

    # Step 2: Construct the normalized DFT matrix
    dft_mat = np.fft.fft(np.eye(dim)) / np.sqrt(dim)

    # Step 3: Compute the circulant matrix. Since circ_mat is formed from the DFT of a real
    # diagonal matrix, it should be real
    return np.real(np.conj(dft_mat.T) @ diag_mat @ dft_mat)

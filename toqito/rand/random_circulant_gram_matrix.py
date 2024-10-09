"""Generate random circulant Gram matrix."""

import numpy as np


def random_circulant_gram_matrix(dim: int) -> np.ndarray:
    r"""Generate a random circulant Gram matrix of specified dimension.

    A circulant matrix is a square matrix where the elements of each row are identical to the elements of the
    previous row such that the elements in one row are relocated by 1 position (in a cyclic manner) compared
    to the previous row. The eigenvalues and eigenvectors of this matrix are derived from the Discrete
    Fourier Transform (DFT).

    For more information on circulant matrices, see :cite:`WikiCirculantMat`. This function utilizes the
    normalized DFT, a variation of DFT with normalized basis vectors.

    For additional information, see :cite:`DSPNormDFT`.

    The function creates a circulant matrix from a random diagonal matrix and the normalized DFT matrix.
    First, it generates a diagonal matrix with random non-negative entries. Next, it constructs the
    normalized DFT matrix. Finally, it computes the circulant matrix, which is real due to its origin
    from the DFT of a real diagonal matrix.

    Examples
    =========
    Generate a random circulant Gram matrix of dimension 4.

    >>> import numpy as np
    >>> from toqito.rand import random_circulant_gram_matrix
    >>> circulant_matrix = random_circulant_gram_matrix(4)
    >>> circulant_matrix.shape
    (4, 4)
    >>> np.allclose(circulant_matrix, circulant_matrix.T)
    True
    >>> circulant_matrix  # doctest: +SKIP
    array([[0.42351891, 0.21058986, 0.04257471, 0.21058986],
           [0.21058986, 0.42351891, 0.21058986, 0.04257471],
           [0.04257471, 0.21058986, 0.42351891, 0.21058986],
           [0.21058986, 0.04257471, 0.21058986, 0.42351891]])


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

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

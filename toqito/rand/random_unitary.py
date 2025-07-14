"""Generates a random unitary matrix."""

import numpy as np


def random_unitary(dim: list[int] | int, is_real: bool = False, seed: int | None = None) -> np.ndarray:
    """Generate a random unitary or orthogonal matrix :footcite:`Ozols_2009_RandU`.

    Calculates a random unitary matrix (if :code:`is_real = False`) or a random real orthogonal
    matrix (if :code:`is_real = True`), uniformly distributed according to the Haar measure.

    Examples
    ==========

    We may generate a random unitary matrix. Here is an example of how we may be able to generate a
    random :math:`2`-dimensional random unitary matrix with complex entries.

    .. jupyter-execute::

     from toqito.rand import random_unitary

     complex_dm = random_unitary(2)

     complex_dm


    We can verify that this is in fact a valid unitary matrix using the :code:`is_unitary` function
    from :code:`|toqito⟩` as follows

    .. jupyter-execute::

     from toqito.matrix_props import is_unitary

     is_unitary(complex_dm)

    We can also generate random unitary matrices that are real-valued as follows.

    .. jupyter-execute::

     from toqito.rand import random_unitary

     real_dm = random_unitary(2, True)

     real_dm


    Again, verifying that this is a valid unitary matrix can be done as follows.

    .. jupyter-execute::

     from toqito.matrix_props import is_unitary

     is_unitary(real_dm)

    We may also generate unitaries such that the dimension argument provided is a :code:`list` as
    opposed to an :code:`int`. Here is an example of a random unitary matrix of dimension :math:`4`.

    .. jupyter-execute::

     from toqito.rand import random_unitary

     mat = random_unitary([4, 4], True)

     mat


    As before, we can verify that this matrix generated is a valid unitary matrix.

    .. jupyter-execute::

     from toqito.matrix_props import is_unitary

     is_unitary(mat)

    It is also possible to pass a seed to this function for reproducibility.

    .. jupyter-execute::

     from toqito.matrix_props import is_unitary

     seeded = random_unitary(2, seed=42)

     seeded

    And once again, we can verify that this matrix generated is a valid unitary matrix.

    .. jupyter-execute::

     from toqito.matrix_props import is_unitary

     is_unitary(seeded)

    References
    ==========
    .. footbibliography::



    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is :code:`False`.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A :code:`dim`-by-:code:`dim` random unitary matrix.

    """
    gen = np.random.default_rng(seed=seed)

    if isinstance(dim, int):
        dim = [dim, dim]

    if dim[0] != dim[1]:
        raise ValueError("Unitary matrix must be square.")

    # Construct the Ginibre ensemble.
    gin = gen.standard_normal((dim[0], dim[1]))

    if not is_real:
        gin = gin + 1j * gen.standard_normal((dim[0], dim[1]))

    # QR decomposition of the Ginibre ensemble.
    q_mat, r_mat = np.linalg.qr(gin)

    # Compute U from QR decomposition.
    r_mat = np.sign(np.diag(r_mat))

    # Protect against potentially zero diagonal entries.
    r_mat[r_mat == 0] = 1

    return q_mat @ np.diag(r_mat)

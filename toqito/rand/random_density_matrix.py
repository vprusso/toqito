"""Generates a random density matrix."""

import numpy as np

from toqito.rand import random_unitary


def random_density_matrix(
    dim: int,
    is_real: bool = False,
    k_param: list[int] | int | None = None,
    distance_metric: str = "haar",
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a random density matrix.

    Generates a random `dim`-by-`dim` density matrix distributed according to the Hilbert-Schmidt measure.
    The matrix is of rank <= `k_param` distributed according to the distribution `distance_metric` If
    `is_real = True`, then all of its entries will be real. The variable `distance_metric` must be one of:

    - `haar` (default):
        Generate a larger pure state according to the Haar measure and trace out the extra dimensions. Sometimes
        called the Hilbert-Schmidt measure when `k_param = dim`.

    - `bures`:
        The Bures measure.

    Examples:
    Using `|toqito⟩`, we may generate a random complex-valued \(n\)- dimensional density matrix. For
    \(d=2\), this can be accomplished as follows.

    ```python exec="1" source="above" session="complex_dm_example"
    from toqito.rand import random_density_matrix

    complex_dm = random_density_matrix(2)

    print(complex_dm)
    ```


    We can verify that this is in fact a valid density matrix using the `is_density` function from
    `|toqito⟩` as follows

    ```python exec="1" source="above" session="complex_dm_example"
    from toqito.matrix_props import is_density

    print(is_density(complex_dm))
    ```


    We can also generate random density matrices that are real-valued as follows.

    ```python exec="1" source="above" session="real_dm_example"
    from toqito.rand import random_density_matrix

    real_dm = random_density_matrix(2, is_real=True)

    print(real_dm)
    ```



    Again, verifying that this is a valid density matrix can be done as follows.

    ```python exec="1" source="above" session="real_dm_example"
    from toqito.matrix_props import is_density

    print(is_density(real_dm))
    ```

    By default, the random density operators are constructed using the Haar measure. We can select to generate the
    random density matrix according to the Bures metric instead as follows.

    ```python exec="1" source="above" session="bures_dm_example"
    from toqito.rand import random_density_matrix

    bures_mat = random_density_matrix(2, distance_metric="bures")

    print(bures_mat)
    ```


    As before, we can verify that this matrix generated is a valid density matrix.

    ```python exec="1" source="above" session="bures_dm_example"
    from toqito.matrix_props import is_density

    print(is_density(bures_mat))
    ```

    It is also possible to pass a seed to this function for reproducibility.
    ```python exec="1" source="above" session="seeded_dm_example"
    from toqito.rand import random_density_matrix

    seeded = random_density_matrix(2, seed=42)

    print(seeded)
    ```

    We can once again verify that this is in fact a valid density matrix using the
    `is_density` function from `|toqito⟩` as follows

    ```python exec="1" source="above" session="seeded_dm_example"
    from toqito.matrix_props import is_density

    seeded = random_density_matrix(2, seed=42)

    print(is_density(seeded))
    ```



    Args:
        dim: The number of rows (and columns) of the density matrix.
        is_real: Boolean denoting whether the returned matrix will have all real entries or not.
        k_param: Default value is equal to `dim`.
        distance_metric: The distance metric used to randomly generate the density matrix. This metric is either the
        Haar measure or the Bures measure. Default value is to use the Haar measure.
        seed: A seed used to instantiate numpy's random number generator.

    Returns:
        A `dim`-by-`dim` random density matrix.

    """
    gen = np.random.default_rng(seed=seed)
    if k_param is None:
        k_param = dim

    # Haar / Hilbert-Schmidt measure.
    gin = gen.random((dim, k_param))

    if not is_real:
        gin = gin + 1j * gen.standard_normal((dim, k_param))

    if distance_metric == "bures":
        gin = random_unitary(dim, is_real, seed=seed) + np.identity(dim) @ gin

    rho = gin @ gin.conj().T

    return np.divide(rho, np.trace(rho))

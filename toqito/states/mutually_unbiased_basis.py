"""Construct a set of mutually unbiased bases."""

import numpy as np
from sympy import isprime, primerange

from toqito.matrices import gen_pauli


def mutually_unbiased_basis(dim: int) -> list[np.ndarray]:
    r"""Generate list of MUBs for a given dimension :cite:`WikiMUB`.

    Note that this function only works if the dimension provided is prime or a power of a prime. Otherwise, we don't
    know how to generate general MUBs.

    Examples
    ========

    For the case of dimension 2, the three mutually unbiased bases are provided by:

    .. math::
        M_0 = \left\{|0\rangle, |1\rangle \right\}, \\
        M_1 = \left\{\frac{|0\rangle + |1\rangle}{\sqrt{2}}, \frac{|0\rangle - |1\rangle}{\sqrt{2}}\right\}
        M_2 = \left\{\frac{|0\rangle + i|1\rangle}{\sqrt{2}}, \frac{|0\rangle - i|1\rangle}{\sqrt{2}}\right\}

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param dim: The dimension of the mutually unbiased bases to produce.
    :return: The set of mutually unbiased bases of dimension :code:`dim` (if known).

    """
    # The first basis will always be the standard basis:
    mats = [np.eye(dim)]

    pauli_x = gen_pauli(1, 0, dim)
    if isprime(dim):
        pauli_z = gen_pauli(0, 1, dim)

        for j in range(dim, 0, -1):
            _, eigen_vec = np.linalg.eig(pauli_x @ pauli_z ** (j))
            mats.append(eigen_vec)
    elif _is_prime_power(dim) and not isprime(dim):
        raise ValueError(f"Dimension {dim} is a prime power but not prime (more complicated no support at the moment).")
    else:
        raise ValueError(f"No general construction of MUBs is known for dimension: {dim}.")

    mubs: list[np.ndarray] = []
    for mat in mats:
        nrows, ncols = mat.shape[0], mat.shape[1]
        for row in range(nrows):
            mub: list[np.ndarray] = []
            for col in range(ncols):
                mub.append(mat[col][row])
            mubs.append(np.array(mub))
    return mubs


def _is_prime_power(n: int) -> bool:
    """Determine if a number is a prime power.

    A number is a prime power if it can be written as p^k, where p is a prime number and k is an integer greater than 0.

    :param n: An integer to check for being a prime power.
    :return: True if n is a prime power, False otherwise.
    """
    # 1 is not considered a prime power
    if n == 1:
        return False

    # Iterate over primes using a generator
    for p in primerange(2, int(n**0.5) + 1):
        # If p is a divisor of n
        if n % p == 0:
            # Keep dividing n by p as long as possible
            while n % p == 0:
                n //= p
            # If n becomes 1, then it's a prime power
            return n == 1
    # If n is itself a prime number
    return isprime(n)

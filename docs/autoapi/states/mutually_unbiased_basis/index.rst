states.mutually_unbiased_basis
==============================

.. py:module:: states.mutually_unbiased_basis

.. autoapi-nested-parse::

   Mutually unbiased basis states.

   If a system prepared in an eigenstate of one of the bases gives an equal probability of (1/d) when measured with respect
   to the other bases, mutually unbiased basis states are orthonormal bases in the Hilbert space Cᵈ.



Functions
---------

.. autoapisummary::

   states.mutually_unbiased_basis.mutually_unbiased_basis
   states.mutually_unbiased_basis._is_prime_power


Module Contents
---------------

.. py:function:: mutually_unbiased_basis(dim)

   Generate list of MUBs for a given dimension :cite:`WikiMUB`.

   Note that this function only works if the dimension provided is prime or a power of a prime. Otherwise, we don't
   know how to generate general MUBs.

   .. rubric:: Examples

   For the case of dimension 2, the three mutually unbiased bases are provided by:

   .. math::
       M_0 = \left\{|0\rangle, |1\rangle \right\}, \\
       M_1 = \left\{\frac{|0\rangle + |1\rangle}{\sqrt{2}}, \frac{|0\rangle - |1\rangle}{\sqrt{2}}\right\}
       M_2 = \left\{\frac{|0\rangle + i|1\rangle}{\sqrt{2}}, \frac{|0\rangle - i|1\rangle}{\sqrt{2}}\right\}

   The six vectors above are obtained accordingly:

   >>> from toqito.states import mutually_unbiased_basis
   >>> mubs = mutually_unbiased_basis(2)
   >>> len(mubs)
   6
   >>> [vec.shape for vec in mubs]
   [(2,), (2,), (2,), (2,), (2,), (2,)]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: The dimension of the mutually unbiased bases to produce.
   :return: The set of mutually unbiased bases of dimension :code:`dim` (if known).


.. py:function:: _is_prime_power(n)

   Determine if a number is a prime power.

   A number is a prime power if it can be written as p^k, where p is a prime number and k is an integer greater than 0.

   :param n: An integer to check for being a prime power.
   :return: True if n is a prime power, False otherwise.



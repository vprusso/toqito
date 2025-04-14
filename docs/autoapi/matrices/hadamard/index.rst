matrices.hadamard
=================

.. py:module:: matrices.hadamard

.. autoapi-nested-parse::

   Generates a Hadamard matrix.



Functions
---------

.. autoapisummary::

   matrices.hadamard.hadamard
   matrices.hadamard._hamming_distance


Module Contents
---------------

.. py:function:: hadamard(n_param = 1)

   Produce a :code:`2^{n_param}` dimensional Hadamard matrix :cite:`WikiHadamard`.

   The standard Hadamard matrix that is often used in quantum information as a
   two-qubit quantum gate is defined as

   .. math::
       H_1 = \frac{1}{\sqrt{2}} \begin{pmatrix}
                                   1 & 1 \\
                                   1 & -1
                                \end{pmatrix}

   In general, the Hadamard matrix of dimension :code:`2^{n_param}` may be
   defined as

   .. math::
       \left( H_n \right)_{i, j} = \frac{1}{2^{\frac{n}{2}}}
       \left(-1\right)^{i \dot j}

   .. rubric:: Examples

   The standard 2-qubit Hadamard matrix can be generated in :code:`|toqito⟩` as

   >>> from toqito.matrices import hadamard
   >>> hadamard(1)
   array([[ 0.70710678,  0.70710678],
          [ 0.70710678, -0.70710678]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param n_param: A non-negative integer (default = 1).
   :return: The Hadamard matrix of dimension :code:`2^{n_param}`.



.. py:function:: _hamming_distance(x_param)

   Calculate the bit-wise Hamming distance of :code:`x_param` from 0.

   The Hamming distance is the number 1s in the integer :code:`x_param`.

   :param x_param: A non-negative integer.
   :return: The hamming distance of :code:`x_param` from 0.



matrix_ops.tensor_comb
======================

.. py:module:: matrix_ops.tensor_comb

.. autoapi-nested-parse::

   Compute tensor combination of list of vectors.



Functions
---------

.. autoapisummary::

   matrix_ops.tensor_comb.tensor_comb


Module Contents
---------------

.. py:function:: tensor_comb(states, k)

   Generate all possible tensor product combinations of quantum states (vectors).

   This function creates a tensor product of quantum state vectors by generating all possible sequences of length `k`
   from a given list of quantum states, and computing the tensor product for each sequence.

   Given `n` quantum states, this function generates `n^k` combinations of sequences of length `k`, computes the tensor
   product for each sequence, and converts each tensor product to its corresponding density matrix.

   For one definition and usage of a quantum sequence, refer to :cite:`Gupta_2024_Optimal`.

   .. rubric:: Examples

   Consider the following basis vectors for a 2-dimensional quantum system.

   .. math::
       e_0 = \left[1, 0 \right]^{\text{T}}, e_1 = \left[0, 1 \right]^{\text{T}}.

   We can generate all possible tensor products for sequences of length 2.

   >>> from toqito.matrix_ops import tensor_comb
   >>> import numpy as np
   >>> e_0 = np.array([1, 0])
   >>> e_1 = np.array([0, 1])
   >>> tensor_comb([e_0, e_1], 2)
   {(0, 0): array([[1, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]), (0, 1): array([[0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]), (1, 0): array([[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 0]]), (1, 1): array([[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1]])}

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If the input list of states is empty.
   :param states: A list of quantum state vectors represented as numpy arrays.
   :param k: The length of the sequence for generating tensor products.
   :return: A dictionary where:

       - Keys represent sequences (as tuples) of quantum state indices,
       - Values are density matrices corresponding to the tensor product of
         the state vectors for the sequence.



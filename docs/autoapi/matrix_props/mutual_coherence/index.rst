matrix_props.mutual_coherence
=============================

.. py:module:: matrix_props.mutual_coherence

.. autoapi-nested-parse::

   Computes the mutual coherence of the columns of a matrix or a list of 1D numpy arrays.



Functions
---------

.. autoapisummary::

   matrix_props.mutual_coherence.mutual_coherence


Module Contents
---------------

.. py:function:: mutual_coherence(matrix)

   Calculate the mutual coherence of a set of states.

   The mutual coherence of a matrix is defined as the maximum absolute value
   of the inner product between any two distinct columns, divided
   by the product of their norms. The mutual coherence is a measure of how
   distinct the given columns are.

   Note: As mutual coherence is also useful in the context of quantum states,
   a list of 1D numpy arrays is also accepted as input.

   .. rubric:: Examples

   >>> import numpy as np
   >>> from toqito.matrix_props.mutual_coherence import mutual_coherence
   >>> matrix_A = np.array([[1, 0], [0, 1]])
   >>> mutual_coherence(matrix_A)
   0.0

   >>> # An example with a larger matrix
   >>> matrix_B = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
   >>> mutual_coherence(matrix_B)
   1/2

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param states: A 2D numpy array or a list of 1D
           numpy arrays representing a set of states (list[np.ndarray] or np.ndarray).
   :raises isinstance: Check if input is valid.
   :return: The mutual coherence.



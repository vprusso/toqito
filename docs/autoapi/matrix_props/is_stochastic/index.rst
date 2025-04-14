matrix_props.is_stochastic
==========================

.. py:module:: matrix_props.is_stochastic

.. autoapi-nested-parse::

   Checks if the matrix is stochastic.



Functions
---------

.. autoapisummary::

   matrix_props.is_stochastic.is_stochastic


Module Contents
---------------

.. py:function:: is_stochastic(mat, mat_type)

   Verify matrix is doubly, right or left stochastic.

   When the nonnegative elements in a row of a square matrix sum up to 1, the matrix is right stochastic and if the
   columns sum up to 1, the matrix is left stochastic :cite:`WikiStochasticMatrix`.

   When a matrix is right and left stochastic, it is a doubly stochastic matrix. :cite:`WikiDoublyStochasticMatrix`.

   .. seealso:: :func:`.is_doubly_stochastic`

   .. rubric:: Examples

   The elements of an identity matrix and a Pauli-X matrix are nonnegative such that the rows and columns sum up to 1.
   We expect these matrices to be left and right stochastic. The same cannot be said about a Pauli-Z or a Pauli-Y
   matrix.

   >>> import numpy as np
   >>> from toqito.matrix_props import is_stochastic
   >>> is_stochastic(np.eye(5), "right")
   True
   >>> is_stochastic(np.eye(5), "left")
   True
   >>> is_stochastic(np.eye(5), "doubly")
   True

   >>> from toqito.matrices import pauli
   >>> from toqito.matrix_props import is_stochastic
   >>> is_stochastic(pauli("X"), "left")
   True
   >>> is_stochastic(pauli("X"), "right")
   True
   >>> is_stochastic(pauli("X"), "doubly")
   True

   >>> from toqito.matrices import pauli
   >>> from toqito.matrix_props import is_stochastic
   >>> is_stochastic(pauli("Z"), "right")
   False
   >>> is_stochastic(pauli("Z"), "left")
   False
   >>> is_stochastic(pauli("Z"), "doubly")
   False

   .. rubric:: References

   .. bibliography::
         :filter: docname in docnames

   :param mat: Matrix of interest
   :param mat_type: Type of stochastic matrix.
                  :code:`"left"` for left stochastic matrix and :code:`"right"` for right stochastic matrix
                  and :code:`"doubly"` for a doubly stochastic matrix.
   :return: Returns :code:`True` if the matrix is doubly, right or left stochastic, :code:`False` otherwise.
   :raises TypeError: If something other than :code:`"doubly"`, :code:`"left"` or :code:`"right"` is used for
                     :code:`mat_type`



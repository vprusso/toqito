"""Check is a matrix is stochastic."""

import numpy as np

from toqito.matrix_props import is_nonnegative, is_square


# ignore the entire file from the coverage report because covered lines erroneously show up as uncovered in the
# report
def is_stochastic(mat: np.ndarray, mat_type: str) -> bool: # pragma: no cover
   r"""Verify matrix is doubly, right or left stochastic.

   When the nonnegative elements in a row of a square matrix sum up to 1, the matrix is right stochastic and if the
   columns sum up to 1, the matrix is left stochastic :cite:`WikiStochasticMatrix`.

   When a matrix is right and left stochastic, it is a doubly stochastic matrix. :cite:`WikiDoublyStochasticMatrix`.

   See Also
   ========
   is_doubly_stochastic

   Examples
   ========
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




   References
   ==========
   .. bibliography::
         :filter: docname in docnames

   :param mat: Matrix of interest
   :param mat_type: Type of stochastic matrix.
                  :code:`"left"` for left stochastic matrix and :code:`"right"` for right stochastic matrix
                  and :code:`"doubly"` for a doubly stochastic matrix.
   :return: Returns :code:`True` if the matrix is doubly, right or left stochastic, :code:`False` otherwise.
   :raises TypeError: If something other than :code:`"doubly"`, :code:`"left"` or :code:`"right"` is used for
                     :code:`mat_type`

   """
   if mat_type == "left":
      axis_num = [0]
   elif mat_type == "right":
      axis_num = [1]
   elif mat_type == "doubly":
      axis_num = [0, 1]
   else:
      raise TypeError("Allowed stochastic matrix types are: left, right, and doubly.")

   if is_square(mat) and is_nonnegative(mat):
      for i in axis_num:
            if np.all(np.sum(mat, i)==1.0):
               return True
            return False
   return False

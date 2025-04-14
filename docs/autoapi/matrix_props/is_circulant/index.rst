matrix_props.is_circulant
=========================

.. py:module:: matrix_props.is_circulant

.. autoapi-nested-parse::

   Checks if the matrix is circulant.



Functions
---------

.. autoapisummary::

   matrix_props.is_circulant.is_circulant


Module Contents
---------------

.. py:function:: is_circulant(mat)

   Determine if matrix is circulant :cite:`WikiCirc`.

   A circulant matrix is a square matrix in which all row vectors are composed
   of the same elements and each row vector is rotated one element to the right
   relative to the preceding row vector.

   .. rubric:: Examples

   Consider the following matrix:

   .. math::
       C = \begin{pmatrix}
               4 & 1 & 2 & 3 \\
               3 & 4 & 1 & 2 \\
               2 & 3 & 4 & 1 \\
               1 & 2 & 3 & 4
           \end{pmatrix}

   As can be seen, this matrix is circulant. We can verify this in
   :code:`|toqito⟩` as

   >>> from toqito.matrix_props import is_circulant
   >>> import numpy as np
   >>> mat = np.array([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]])
   >>> is_circulant(mat)
   True

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check the circulancy of.
   :return: Return `True` if :code:`mat` is circulant; `False` otherwise.




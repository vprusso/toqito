matrix_props.is_nonnegative
===========================

.. py:module:: matrix_props.is_nonnegative

.. autoapi-nested-parse::

   Checks if the matrix is nonnegative or doubly nonnegative.



Functions
---------

.. autoapisummary::

   matrix_props.is_nonnegative.is_nonnegative


Module Contents
---------------

.. py:function:: is_nonnegative(input_mat, mat_type = 'nonnegative')

   Check if the matrix is nonnegative.

   When all the entries in the matrix are larger than or equal to zero the matrix of interest is a
   nonnegative matrix :cite:`WikiNonNegative`.

   When a matrix is nonegative and positive semidefinite :cite:`WikiPosDef`, the matrix is doubly nonnegative.


   .. rubric:: Examples

   We expect an identity matrix to be nonnegative.

   >>> import numpy as np
   >>> from toqito.matrix_props import is_nonnegative
   >>> is_nonnegative(np.eye(2))
   True
   >>> is_nonnegative(np.eye(2), "doubly")
   True
   >>> is_nonnegative(np.array([[1, -1], [1, 1]]))
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param input_mat: np.ndarray
                   Matrix of interest.
   :param mat_type: Type of nonnegative matrix. :code:`"nonnegative"` for a nonnegative matrix and :code:`"doubly"`
                   for a doubly nonnegative matrix.
   :raises TypeError: If something other than :code:`"doubly"`or :code:`"nonnegative"` is used for :code:`mat_type`.



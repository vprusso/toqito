matrix_props.is_block_positive
==============================

.. py:module:: matrix_props.is_block_positive

.. autoapi-nested-parse::

   Checks if the matrix is block positive.



Functions
---------

.. autoapisummary::

   matrix_props.is_block_positive.is_block_positive


Module Contents
---------------

.. py:function:: is_block_positive(mat, k = 1, dim = None, effort = 2, rtol = 1e-05)

   Check if matrix is block positive :cite:`Johnston_2012_Norms`.

   .. rubric:: Examples

   The swap operator is always block positive, since it is the Choi
   matrix of the transpose map.

   >>> from toqito.matrix_props.is_block_positive import is_block_positive
   >>> from toqito.perms.swap_operator import swap_operator
   >>>
   >>> mat = swap_operator(3)
   >>> is_block_positive(mat)
   True

   However, it's not 2 - block positive.

   >>> from toqito.matrix_props.is_block_positive import is_block_positive
   >>> from toqito.perms.swap_operator import swap_operator
   >>>
   >>> mat = swap_operator(3)
   >>> is_block_positive(mat, k=2)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises RuntimeError: Unable to determine k-block positivity. Please consider increasing the relative tolerance or
                           the effort level.
   :param mat: A bipartite Hermitian operator.
   :param k: A positive integer indicating that the function should determine whether or not
             the input operator is k-block positive, i.e., whether or not it remains nonnegative
             under left and right multiplication by vectors with Schmidt rank <= k (default 1).
   :param dim: The dimension of the two sub-systems. By default it's assumed to be equal.
   :param effort: An integer value indicating the amount of computation you want to devote to
                  determine block positivity before giving up.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :return: Return :code:`True` if matrix is k-block positive definite,
            :code:`False` if not, or raise a runtime error if we are unable to determine
            whether or not the operator is block positive.




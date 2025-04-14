matrices.standard_basis
=======================

.. py:module:: matrices.standard_basis

.. autoapi-nested-parse::

   Constructs the standard basis.



Functions
---------

.. autoapisummary::

   matrices.standard_basis.standard_basis


Module Contents
---------------

.. py:function:: standard_basis(dim, flatten = False)

   Create standard basis of dimension :code:`dim`.

   Create a list containing the elements of the standard basis for the
   given dimension:

   .. math::

       |1> = (1, 0, 0, ..., 0)^T
       |2> = (0, 1, 0, ..., 0)^T
       .
       .
       .
       |n> = (0, 0, 0, ..., 1)^T

   This function was inspired by :cite:`Seshadri_2021_Git, Seshadri_2021_Theory, Seshadri_2021_Versatile`

   .. rubric:: Examples

   >>> from toqito.matrices import standard_basis
   >>> standard_basis(2)
   [array([[1.],
          [0.]]), array([[0.],
          [1.]])]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: The dimension of the basis.
   :param flatten: If True, the basis is returned as a flattened list.
   :return: A list of numpy.ndarray of shape (n, 1).



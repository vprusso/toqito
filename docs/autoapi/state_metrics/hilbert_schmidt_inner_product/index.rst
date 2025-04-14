state_metrics.hilbert_schmidt_inner_product
===========================================

.. py:module:: state_metrics.hilbert_schmidt_inner_product

.. autoapi-nested-parse::

   Hilbert-Schmidt Inner Product refers to the inner product between two Hilbert-Schmidt operators.



Functions
---------

.. autoapisummary::

   state_metrics.hilbert_schmidt_inner_product.hilbert_schmidt_inner_product


Module Contents
---------------

.. py:function:: hilbert_schmidt_inner_product(a_mat, b_mat)

   Compute the Hilbert-Schmidt inner product between two matrices :cite:`WikiHilbSchOp`.

   The Hilbert-Schmidt inner product between :code:`a_mat` and :code:`b_mat` is defined as

   .. math::

       HS = (A|B) = Tr[A^\dagger B]

   where :math:`|B\rangle = \text{vec}(B)` and :math:`\langle A|` is the dual vector to :math:`|A \rangle`.

   Note: This function has been adapted from :cite:`Rigetti_2022_Forest`.

   .. rubric:: Examples

   One may consider taking the Hilbert-Schmidt distance between two Hadamard matrices.

   >>> import numpy as np
   >>> from toqito.matrices import hadamard
   >>> from toqito.state_metrics import hilbert_schmidt_inner_product
   >>> h = hadamard(1)
   >>> np.around(hilbert_schmidt_inner_product(h, h), decimals=2)
   np.float64(2.0)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param a_mat: An input matrix provided as a numpy array.
   :param b_mat: An input matrix provided as a numpy array.
   :return: The Hilbert-Schmidt inner product between :code:`a_mat` and
            :code:`b_mat`.




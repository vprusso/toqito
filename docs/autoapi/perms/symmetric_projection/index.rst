perms.symmetric_projection
==========================

.. py:module:: perms.symmetric_projection

.. autoapi-nested-parse::

   Symmetric projection operator produces a projection onto a symmetric subspace.



Functions
---------

.. autoapisummary::

   perms.symmetric_projection.symmetric_projection


Module Contents
---------------

.. py:function:: symmetric_projection(dim, p_val = 2, partial = False)

   Produce the projection onto the symmetric subspace :cite:`Chen_2014_Symmetric`.

   For a complex Euclidean space :math:`\mathcal{X}` and a positive integer :math:`n`, the projection onto the
   symmetric subspace is given by

   .. math::
       \frac{1}{n!} \sum_{\pi \in S_n} W_{\pi}

   where :math:`W_{\pi}` is the swap operator and where :math:`S_n` is the symmetric group on :math:`n` symbols.

   Produces the orthogonal projection onto the symmetric subspace of :code:`p_val` copies of `dim`-dimensional space.
   If `partial = True`, then the symmetric projection (PS) isn't the orthogonal projection itself, but rather a matrix
   whose columns form an orthonormal basis for the symmetric subspace (and hence the PS * PS' is the orthogonal
   projection onto the symmetric subspace).

   This function was adapted from the QETLAB package.

   .. rubric:: Examples

   The :math:`2`-dimensional symmetric projection with :math:`p=1` is given as :math:`2`-by-:math:`2` identity matrix

   .. math::
       \begin{pmatrix}
           1 & 0 \\
           0 & 1
       \end{pmatrix}.

   Using :code:`|toqito⟩`, we can see this gives the proper result.

   >>> from toqito.perms import symmetric_projection
   >>> symmetric_projection(2, 1)
   array([[1., 0.],
          [0., 1.]])

   When :math:`d = 2` and :math:`p = 2` we have that

   .. math::
       \begin{pmatrix}
           1 & 0 & 0 & 0 \\
           0 & 1/2 & 1/2 & 0 \\
           0 & 1/2 & 1/2 & 0 \\
           0 & 0 & 0 & 1
       \end{pmatrix}.

   Using :code:`|toqito⟩` we can see this gives the proper result.

   >>> from toqito.perms import symmetric_projection
   >>> symmetric_projection(dim=2)
   array([[1. , 0. , 0. , 0. ],
          [0. , 0.5, 0.5, 0. ],
          [0. , 0.5, 0.5, 0. ],
          [0. , 0. , 0. , 1. ]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: The dimension of the local systems.
   :param p_val: Default value of 2.
   :param partial: Default value of 0.
   :return: Projection onto the symmetric subspace.




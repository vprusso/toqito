matrices.gell_mann
==================

.. py:module:: matrices.gell_mann

.. autoapi-nested-parse::

   Generates the Gell-Mann operator matrices.



Functions
---------

.. autoapisummary::

   matrices.gell_mann.gell_mann


Module Contents
---------------

.. py:function:: gell_mann(ind, is_sparse = False)

   Produce a Gell-Mann operator :cite:`WikiGellMann`.

   Generates the 3-by-3 Gell-Mann matrix indicated by the value of
   :code:`ind`.  When :code:`ind = 0` gives the identity matrix, while values
   1 through 8 each indicate one of the other 8 Gell-Mann matrices.

   The 9 Gell-Mann matrices are defined as follows:

   .. math::
       \begin{equation}
           \begin{aligned}
               \lambda_0 = \begin{pmatrix}
                               1 & 0 & 0 \\
                               0 & 1 & 0 \\
                               0 & 0 & 1
                           \end{pmatrix}, \quad
               \lambda_1 = \begin{pmatrix}
                               0 & 1 & 0 \\
                               1 & 0 & 0 \\
                               0 & 0 & 0
                           \end{pmatrix}, \quad &
               \lambda_2 = \begin{pmatrix}
                               0 & -i & 0 \\
                               i & 0 & 0 \\
                               0 & 0 & 0
                           \end{pmatrix},  \\
               \lambda_3 = \begin{pmatrix}
                               1 & 0 & 0 \\
                               0 & -1 & 0 \\
                               0 & 0 & 0
                           \end{pmatrix}, \quad
               \lambda_4 = \begin{pmatrix}
                               0 & 0 & 1 \\
                               0 & 0 & 0 \\
                               1 & 0 & 0
                           \end{pmatrix}, \quad &
               \lambda_5 = \begin{pmatrix}
                               0 & 0 & -i \\
                               0 & 0 & 0 \\
                               i & 0 & 0
                           \end{pmatrix},  \\
               \lambda_6 = \begin{pmatrix}
                               0 & 0 & 0 \\
                               0 & 0 & 1 \\
                               0 & 1 & 0
                           \end{pmatrix}, \quad
               \lambda_7 = \begin{pmatrix}
                               0 & 0 & 0 \\
                               0 & 0 & -i \\
                               0 & i & 0
                           \end{pmatrix}, \quad &
               \lambda_8 = \frac{1}{\sqrt{3}} \begin{pmatrix}
                                                   1 & 0 & 0 \\
                                                   0 & 1 & 0 \\
                                                   0 & 0 & -2
                                               \end{pmatrix}.
               \end{aligned}
           \end{equation}

   .. rubric:: Examples

   The Gell-Mann matrix generated from :code:`idx = 2` yields the following
   matrix:

   .. math::

       \lambda_2 = \begin{pmatrix}
                           0 & -i & 0 \\
                           i & 0 & 0 \\
                           0 & 0 & 0
                   \end{pmatrix}

   >>> from toqito.matrices import gell_mann
   >>> gell_mann(2)
   array([[ 0.+0.j, -0.-1.j,  0.+0.j],
          [ 0.+1.j,  0.+0.j,  0.+0.j],
          [ 0.+0.j,  0.+0.j,  0.+0.j]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: Indices must be integers between 0 and 8.
   :param ind: An integer between 0 and 8 (inclusive).
   :param is_sparse: Boolean to determine whether array is sparse. Default value is :code:`False`.




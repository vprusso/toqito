matrix_props.commutant
======================

.. py:module:: matrix_props.commutant

.. autoapi-nested-parse::

   Module for computing the commutant of a set of matrices.



Functions
---------

.. autoapisummary::

   matrix_props.commutant.commutant


Module Contents
---------------

.. py:function:: commutant(A)

   Compute an orthonormal basis for the commutant algebra :cite:`PlanetMathCommutant`.

   Given a matrix :math:`A` or a set of matrices :math:`\mathcal{A} = \{A_1, A_2, \dots\}`,
   this function determines an orthonormal basis (with respect to the Hilbert-Schmidt inner product)
   for the algebra of matrices that commute with every matrix in :math:`\mathcal{A}`.

   The commutant of a single matrix :math:`A \in \mathbb{C}^{n \times n}` consists of all matrices
   :math:`X \in \mathbb{C}^{n \times n}` satisfying:

   .. math:: A X = X A.

   More generally, for a set of matrices :math:`\mathcal{A} = \{A_1, A_2, \dots\}`, the commutant
   consists of all matrices :math:`X` satisfying:

   .. math:: A_i X = X A_i \quad \forall A_i \in \mathcal{A}.

   This condition can be rewritten in vectorized form as:

   .. math::
       (A_i \otimes I - I \otimes A_i^T) \text{vec}(X) = 0, \quad \forall A_i \in \mathcal{A}.

   where :math:`\text{vec}(X)` denotes the column-wise vectorization of :math:`X`.
   The null space of this equation provides a basis for the commutant.

   This implementation is based on :cite:`QETLAB_link`.

   .. rubric:: Examples

   Consider the following set of matrices:

   .. math::
       A_1 = \begin{pmatrix}
               1 & 0 \\
               0 & -1
           \end{pmatrix}, \quad
       A_2 = \begin{pmatrix}
               0 & 1 \\
               1 & 0
           \end{pmatrix}

   The commutant consists of matrices that commute with both :math:`A_1` and :math:`A_2`.

   >>> import numpy as np
   >>> from toqito.matrix_props import commutant
   >>>
   >>> A1 = np.array([[1, 0], [0, -1]])
   >>> A2 = np.array([[0, 1], [1, 0]])
   >>> basis = commutant([A1, A2])
   >>> basis
   [array([[0.70710678, 0.        ],
          [0.        , 0.70710678]])]

   Now, consider a single matrix:

   .. math::
       A = \begin{pmatrix}
               1 & 1 \\
               0 & 1
           \end{pmatrix}

   >>> A = np.array([[1, 1], [0, 1]])
   >>> basis = commutant(A)
   >>> basis
   [array([[0.70710678, 0.        ],
          [0.        , 0.70710678]]), array([[ 0., -1.],
          [ 0.,  0.]])]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param A: A single matrix of the form np.ndarray or a list of square matrices of the same dimension.
   :return: A list of matrices forming an orthonormal basis for the commutant.



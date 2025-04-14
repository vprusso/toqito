matrices.gen_pauli_x
====================

.. py:module:: matrices.gen_pauli_x

.. autoapi-nested-parse::

   Produces a generalized Pauli-X operator matrix.



Functions
---------

.. autoapisummary::

   matrices.gen_pauli_x.gen_pauli_x


Module Contents
---------------

.. py:function:: gen_pauli_x(dim)

   Produce a :code:`dim`-by-:code:`dim` gen_pauli_x matrix :cite:`WikiPauliGen`.

   Returns the gen_pauli_x matrix of dimension :code:`dim` described in :cite:`WikiPauliGen`.
   The gen_pauli_x matrix generates the following :code:`dim`-by-:code:`dim` matrix:

   .. math::
       \Sigma_{1, d} = \begin{pmatrix}
                       0 & 0 & 0 & \ldots & 0 & 1 \\
                       1 & 0 & 0 & \ldots & 0 & 0 \\
                       0 & 1 & 0 & \ldots & 0 & 0 \\
                       0 & 0 & 1 & \ldots & 0 & 0 \\
                       \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                       0 & 0 & 0 & \ldots & 1 & 0
                   \end{pmatrix}

   The gen_pauli_x matrix is primarily used in the construction of the generalized
   Pauli operators.

   .. rubric:: Examples

   The gen_pauli_x matrix generated from :math:`d = 3` yields the following matrix:

   .. math::
       \Sigma_{1, 3} =
       \begin{pmatrix}
           0 & 0 & 1 \\
           1 & 0 & 0 \\
           0 & 1 & 0
       \end{pmatrix}

   >>> from toqito.matrices import gen_pauli_x
   >>> gen_pauli_x(3)
   array([[0., 0., 1.],
          [1., 0., 0.],
          [0., 1., 0.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: Dimension of the matrix.
   :return: :code:`dim`-by-:code:`dim` gen_pauli_x matrix.




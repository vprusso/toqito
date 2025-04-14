matrices.gen_pauli_z
====================

.. py:module:: matrices.gen_pauli_z

.. autoapi-nested-parse::

   Produces a generalized Pauli-Z operator matrix.



Functions
---------

.. autoapisummary::

   matrices.gen_pauli_z.gen_pauli_z


Module Contents
---------------

.. py:function:: gen_pauli_z(dim)

   Produce gen_pauli_z matrix :cite:`WikiClock`.

   Returns the gen_pauli_z matrix of dimension :code:`dim` described in :cite:`WikiClock`.
   The gen_pauli_z matrix generates the following :code:`dim`-by-:code:`dim` matrix

   .. math::
       \Sigma_{1, d} = \begin{pmatrix}
                       1 & 0 & 0 & \ldots & 0 \\
                       0 & \omega & 0 & \ldots & 0 \\
                       0 & 0 & \omega^2 & \ldots & 0 \\
                       \vdots & \vdots & \vdots & \ddots & \vdots \\
                       0 & 0 & 0 & \ldots & \omega^{d-1}
                  \end{pmatrix}

   where :math:`\omega` is the n-th primitive root of unity.

   The gen_pauli_z matrix is primarily used in the construction of the generalized
   Pauli operators.

   .. rubric:: Examples

   The gen_pauli_z matrix generated from :math:`d = 3` yields the following matrix:

   .. math::
       \Sigma_{1, 3} = \begin{pmatrix}
           1 & 0 & 0 \\
           0 & \omega & 0 \\
           0 & 0 & \omega^2
       \end{pmatrix}

   >>> from toqito.matrices import gen_pauli_z
   >>> gen_pauli_z(3)
   array([[ 1. +0.j       ,  0. +0.j       ,  0. +0.j       ],
          [ 0. +0.j       , -0.5+0.8660254j,  0. +0.j       ],
          [ 0. +0.j       ,  0. +0.j       , -0.5-0.8660254j]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: Dimension of the matrix.
   :return: :code:`dim`-by-:code:`dim` gen_pauli_z matrix.




matrices.cnot
=============

.. py:module:: matrices.cnot

.. autoapi-nested-parse::

   CNOT matrix generates the CNOT operator matrix.



Functions
---------

.. autoapisummary::

   matrices.cnot.cnot


Module Contents
---------------

.. py:function:: cnot()

   Produce the CNOT matrix :cite:`WikiCNOT`.

   The CNOT matrix is defined as

   .. math::
       \text{CNOT} =
       \begin{pmatrix}
           1 & 0 & 0 & 0 \\
           0 & 1 & 0 & 0 \\
           0 & 0 & 0 & 1 \\
           0 & 0 & 1 & 0
       \end{pmatrix}.

   .. rubric:: Examples

   >>> from toqito.matrices import cnot
   >>> cnot()
   array([[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :return: The CNOT matrix.




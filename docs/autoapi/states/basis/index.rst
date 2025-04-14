states.basis
============

.. py:module:: states.basis

.. autoapi-nested-parse::

   Basis state represent the standard basis vectors of some n-dimensional Hilbert Space.

   Here, n can be given as a parameter as shown below.



Functions
---------

.. autoapisummary::

   states.basis.basis


Module Contents
---------------

.. py:function:: basis(dim, pos)

   Obtain the ket of dimension :code:`dim` :cite:`WikiBraKet`.

   .. rubric:: Examples

   The standard basis ket vectors given as :math:`|0 \rangle` and :math:`|1 \rangle` where

   .. math::
       |0 \rangle = \left[1, 0 \right]^{\text{T}} \quad \text{and} \quad
       |1 \rangle = \left[0, 1 \right]^{\text{T}},

   can be obtained in :code:`|toqito⟩` as follows.

   Example:  Ket basis vector: :math:`|0\rangle`.

   >>> from toqito.states import basis
   >>> basis(2, 0)
   array([[1],
          [0]])

   Example: Ket basis vector: :math:`|1\rangle`.

   >>> from toqito.states import basis
   >>> basis(2, 1)
   array([[0],
          [1]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If the input position is not in the range [0, dim - 1].
   :param dim: The dimension of the column vector.
   :param pos: The position in which to place a 1.
   :return: The column vector of dimension :code:`dim` with all entries set to `0` except the entry
            at position `1`.




state_ops.schmidt_decomposition
===============================

.. py:module:: state_ops.schmidt_decomposition

.. autoapi-nested-parse::

   Schmidt decomposition operation computes the schmidt decomposition of a quantum state or an operator.



Functions
---------

.. autoapisummary::

   state_ops.schmidt_decomposition.schmidt_decomposition
   state_ops.schmidt_decomposition._operator_schmidt_decomposition


Module Contents
---------------

.. py:function:: schmidt_decomposition(rho, dim = None, k_param = 0)

   Compute the Schmidt decomposition of a bipartite vector :cite:`WikiScmidtDecomp`.

   .. rubric:: Examples

   Consider the :math:`3`-dimensional maximally entangled state:

   .. math::
       u = \frac{1}{\sqrt{3}} \left( |000 \rangle + |111 \rangle + |222 \rangle \right).

   We can generate this state using the :code:`|toqito⟩` module as follows.

   >>> from toqito.states import max_entangled
   >>> max_entangled(3)
   array([[0.57735027],
          [0.        ],
          [0.        ],
          [0.        ],
          [0.57735027],
          [0.        ],
          [0.        ],
          [0.        ],
          [0.57735027]])

   Computing the Schmidt decomposition of :math:`u`, we can obtain the corresponding singular
   values of :math:`u` as

   .. math::
       \frac{1}{\sqrt{3}} \left[1, 1, 1 \right]^{\text{T}}.

   >>> from toqito.states import max_entangled
   >>> from toqito.state_ops import schmidt_decomposition
   >>> singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3))
   >>> singular_vals
   array([[0.57735027],
          [0.57735027],
          [0.57735027]])
   >>> u_mat
   array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]])
   >>> vt_mat
   array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices are not of equal dimension.
   :param rho: A bipartite quantum state to compute the Schmidt decomposition of.
   :param dim: An array consisting of the dimensions of the subsystems (default gives subsystems
               equal dimensions).
   :param k_param: How many terms of the Schmidt decomposition should be computed (default is 0).
   :return: The Schmidt decomposition of the :code:`rho` input.



.. py:function:: _operator_schmidt_decomposition(rho, dim = None, k_param = 0)

   Calculate the Schmidt decomposition of an operator (matrix).

   Given an input `rho` provided as a matrix, determine its corresponding
   Schmidt decomposition.

   :raises ValueError: If matrices are not of equal dimension..
   :param rho: The matrix.
   :param dim: The dimension of the matrix
   :param k_param: The number of Schmidt coefficients to compute.
   :return: The Schmidt decomposition of the :code:`rho` input.



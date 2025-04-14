state_props.is_product
======================

.. py:module:: state_props.is_product

.. autoapi-nested-parse::

   Checks if a quantum state is product state.



Functions
---------

.. autoapisummary::

   state_props.is_product.is_product
   state_props.is_product._is_product
   state_props.is_product._operator_is_product


Module Contents
---------------

.. py:function:: is_product(rho, dim = None)

   Determine if a given vector is a product state :cite:`WikiSepSt`.

   If the input is deemed to be product, then the product decomposition is also
   returned.

   .. rubric:: Examples

   Consider the following Bell state

   .. math::
       u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

   The corresponding density matrix of :math:`u` may be calculated by:

   .. math::
       \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                        1 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        1 & 0 & 0 & 1
                      \end{pmatrix} \in \text{D}(\mathcal{X}).

   We can provide the input as either the vector :math:`u` or the denisty matrix :math:`\rho`.
   In either case, this represents an entangled state (and hence a non-product state).

   >>> from toqito.state_props import is_product
   >>> from toqito.states import bell
   >>> rho = bell(0) @ bell(0).conj().T
   >>> u_vec = bell(0)
   >>> is_product(rho)
   (array([False]), None)
   >>>
   >>> is_product(u_vec)
   (array([False]), None)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param rho: The vector or matrix to check.
   :param dim: The dimension of the input.
   :return: :code:`True` if :code:`rho` is a product vector and :code:`False` otherwise.



.. py:function:: _is_product(rho, dim = None)

   Determine if input is a product state recursive helper.

   :param rho: The vector or matrix to check.
   :param dim: The dimension of the input.
   :return: :code:`True` if :code:`rho` is a product vector and :code:`False` otherwise.


.. py:function:: _operator_is_product(rho, dim = None)

   Determine if a given matrix is a product operator.

   Given an input `rho` provided as a matrix, determine if it is a product
   state.
   :param rho: The matrix to check.
   :param dim: The dimension of the matrix
   :return: :code:`True` if :code:`rho` is product and :code:`False` otherwise.



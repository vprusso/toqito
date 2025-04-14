matrix_props.majorizes
======================

.. py:module:: matrix_props.majorizes

.. autoapi-nested-parse::

   Determine if one vector or matrix majorizes another.



Functions
---------

.. autoapisummary::

   matrix_props.majorizes.majorizes


Module Contents
---------------

.. py:function:: majorizes(a_var, b_var)

   Determine if one vector or matrix majorizes another :cite:`WikiMajorization`.

   Given :math:`a, b \in \mathbb{R}^d`, we say that :math:`a` **weakly majorizes** (or dominates)
   :math:`b` from below if and only if

   .. math::
       \sum_{i=1}^k a_i^{\downarrow} \geq \sum_{i=1}^k b_i^{\downarrow}

   for all :math:`k \in \{1, \ldots, d\}`.

   This function was adapted from the QETLAB package.

   .. rubric:: Examples

   Simple example illustrating that the vector :math:`(3, 0, 0)` majorizes the vector
   :math:`(1, 1, 1)`.

   >>> from toqito.matrix_props import majorizes
   >>> majorizes([3, 0, 0], [1, 1, 1])
   True

   The majorization criterion says that every separable state
   :math:`\rho \in \text{D}(\mathcal{A} \otimes \mathcal{B})` is such that
   :math:`\text{Tr}_{\mathcal{B}}(\rho)` majorizes
   :math:`\text{Tr}_{\mathcal{A}}(\rho)`.

   >>> from toqito.matrix_props import majorizes
   >>> from toqito.states import max_entangled
   >>> from toqito.channels import partial_trace
   >>>
   >>> v_vec = max_entangled(3)
   >>> rho = v_vec @ v_vec.conj().T
   >>> majorizes(partial_trace(rho, [1]), rho)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param a_var: Matrix or vector provided as list or np.array.
   :param b_var: Matrix or vector provided as list or np.array.
   :return: Return :code:`True` if :code:`a_var` majorizes :code:`b_var` and :code:`False`
            otherwise.




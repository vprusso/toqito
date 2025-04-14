matrix_props.trace_norm
=======================

.. py:module:: matrix_props.trace_norm

.. autoapi-nested-parse::

   Computes the trace norm metric of a density matrix.



Functions
---------

.. autoapisummary::

   matrix_props.trace_norm.trace_norm


Module Contents
---------------

.. py:function:: trace_norm(rho)

   Compute the trace norm of the state :cite:`Quantiki_TrNorm`.

   Also computes the operator 1-norm when inputting an operator.

   The trace norm :math:`||\rho||_1` of a density matrix :math:`\rho` is the sum of the singular
   values of :math:`\rho`. The singular values are the roots of the eigenvalues of
   :math:`\rho \rho^*`.

   .. rubric:: Examples

   Consider the following Bell state

   .. math::
       u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

   The corresponding density matrix of :math:`u` may be calculated by:

   .. math::
       \rho = u u^* = \begin{pmatrix}
                        1 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        1 & 0 & 0 & 1
                      \end{pmatrix} \in \text{D}(\mathcal{X}).

   It can be observed using :code:`|toqito⟩` that :math:`||\rho||_1 = 1` as follows.

   >>> from toqito.states import bell
   >>> from toqito.matrix_props import trace_norm
   >>> rho = bell(0) @ bell(0).conj().T
   >>> trace_norm(rho)
   np.float64(0.9999999999999999)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param rho: Density operator.
   :return: The trace norm of :code:`rho`.




channel_metrics.completely_bounded_trace_norm
=============================================

.. py:module:: channel_metrics.completely_bounded_trace_norm

.. autoapi-nested-parse::

   Computes the completely bounded trace norm of a quantum channel.



Functions
---------

.. autoapisummary::

   channel_metrics.completely_bounded_trace_norm.completely_bounded_trace_norm


Module Contents
---------------

.. py:function:: completely_bounded_trace_norm(phi, solver = 'cvxopt', **kwargs)

   Find the completely bounded trace norm of a quantum channel.

   Also known as the diamond norm of a quantum
   channel (Section 3.3.2 of :cite:`Watrous_2018_TQI`). The algorithm in p.11 of :cite:`Watrous_2012_Simpler` with
   implementation in QETLAB :cite:`QETLAB_link` is used.

   .. rubric:: Examples

   To computer the completely bounded spectral norm of a depolarizing channel,

   >>> from toqito.channels import depolarizing
   >>> from toqito.channel_metrics import completely_bounded_trace_norm
   >>>
   >>> # Define the depolarizing channel
   >>> choi_depolarizing = depolarizing(dim=2, param_p=0.2)
   >>> completely_bounded_trace_norm(choi_depolarizing)
   1

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrix is not square.
   :param phi: superoperator as choi matrix
   :param solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
   :param kwargs: Additional arguments to pass to picos' solve method.
   :return: The completely bounded trace norm of the channel




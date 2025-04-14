channel_metrics.completely_bounded_spectral_norm
================================================

.. py:module:: channel_metrics.completely_bounded_spectral_norm

.. autoapi-nested-parse::

   Computes the completely bounded spectral norm of a quantum channel.



Functions
---------

.. autoapisummary::

   channel_metrics.completely_bounded_spectral_norm.completely_bounded_spectral_norm


Module Contents
---------------

.. py:function:: completely_bounded_spectral_norm(phi)

   Compute the completely bounded spectral norm of a quantum channel.

   As defined in :cite:`Watrous_2009_Semidefinite` and :cite:`QETLAB_link`.

   .. rubric:: Examples

   To computer the completely bounded spectral norm of a depolarizing channel,

   >>> from toqito.channels import depolarizing
   >>> from toqito.channel_metrics import completely_bounded_spectral_norm
   >>>
   >>> # Define the depolarizing channel
   >>> choi_depolarizing = depolarizing(dim=2, param_p=0.2)
   >>> completely_bounded_spectral_norm(choi_depolarizing)
   1

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: superoperator
   :return: The completely bounded spectral norm of the channel



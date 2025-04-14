state_props.l1_norm_coherence
=============================

.. py:module:: state_props.l1_norm_coherence

.. autoapi-nested-parse::

   Computes the l1-norm of coherence of a quantum state.



Functions
---------

.. autoapisummary::

   state_props.l1_norm_coherence.l1_norm_coherence


Module Contents
---------------

.. py:function:: l1_norm_coherence(rho)

   Compute the l1-norm of coherence of a quantum state :cite:`Rana_2017_Log`.

   The :math:`\ell_1`-norm of coherence of a quantum state :math:`\rho` is
   defined as

   .. math::
       C_{\ell_1}(\rho) = \sum_{i \not= j} \left|\rho_{i,j}\right|,

   where :math:`\rho_{i,j}` is the :math:`(i,j)^{th}`-entry of :math:`\rho`
   in the standard basis.

   The :math:`\ell_1`-norm of coherence is the sum of the absolute values of
   the sum of the absolute values of the off-diagonal entries of the density
   matrix :code:`rho` in the standard basis.

   This function was adapted from QETLAB.

   .. rubric:: Examples

   The largest possible value of the :math:`\ell_1`-norm of coherence on
   :math:`d`-dimensional states is :math:`d-1`, and is attained exactly by
   the "maximally coherent states": pure states whose entries all have the
   same absolute value.

   >>> from toqito.state_props import l1_norm_coherence
   >>> import numpy as np
   >>>
   >>> # Maximally coherent state.
   >>> v = np.ones((3,1))/np.sqrt(3)
   >>> '%.1f' % l1_norm_coherence(v)
   '2.0'

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param rho: A matrix or vector.
   :return: The l1-norm coherence of :code:`rho`.




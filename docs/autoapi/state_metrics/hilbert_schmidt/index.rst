state_metrics.hilbert_schmidt
=============================

.. py:module:: state_metrics.hilbert_schmidt

.. autoapi-nested-parse::

   Hilbert-Schmidt metric is a distance metric used to generate an entanglement measure.



Functions
---------

.. autoapisummary::

   state_metrics.hilbert_schmidt.hilbert_schmidt


Module Contents
---------------

.. py:function:: hilbert_schmidt(rho, sigma)

   Compute the Hilbert-Schmidt distance between two states :cite:`WikiHilbSchOp`.

   The Hilbert-Schmidt distance between density operators :math:`\rho` and :math:`\sigma` is defined as

   .. math::
       D_{\text{HS}}(\rho, \sigma) = \text{Tr}((\rho - \sigma)^2) = \left\lVert \rho - \sigma
       \right\rVert_2^2.

   .. rubric:: Examples

   One may consider taking the Hilbert-Schmidt distance between two Bell states. In :code:`|toqito⟩`,
   one may accomplish this as

   >>> import numpy as np
   >>> from toqito.states import bell
   >>> from toqito.state_metrics import hilbert_schmidt
   >>> rho = bell(0) @ bell(0).conj().T
   >>> sigma = bell(3) @ bell(3).conj().T
   >>> np.around(hilbert_schmidt(rho, sigma), decimals=2)
   np.float64(1.0)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices are not density operators.
   :param rho: An input matrix.
   :param sigma: An input matrix.
   :return: The Hilbert-Schmidt distance between :code:`rho` and :code:`sigma`.




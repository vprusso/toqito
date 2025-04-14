state_metrics.sub_fidelity
==========================

.. py:module:: state_metrics.sub_fidelity

.. autoapi-nested-parse::

   Sub-fidelity metric is a lower bound for the fidelity.

   The sub-fidelity metric is a concave function and sub-multiplicative.



Functions
---------

.. autoapisummary::

   state_metrics.sub_fidelity.sub_fidelity


Module Contents
---------------

.. py:function:: sub_fidelity(rho, sigma)

   Compute the sub fidelity of two density matrices :cite:`Miszczak_2008_Sub`.

   The sub-fidelity is a measure of similarity between density operators. It is defined as

   .. math::
       E(\rho, \sigma) = \text{Tr}(\rho \sigma) +
       \sqrt{2 \left[ \text{Tr}(\rho \sigma)^2 - \text{Tr}(\rho \sigma \rho \sigma) \right]},

   where :math:`\sigma` and :math:`\rho` are density matrices. The sub-fidelity serves as an lower bound for the
   fidelity.

   .. rubric:: Examples

   Consider the following pair of states:

   .. math::
       \rho = \frac{3}{4}|0\rangle \langle 0| +
              \frac{1}{4}|1 \rangle \langle 1|
               \quad \text{and} \quad
       \sigma = \frac{1}{8}|0 \rangle \langle 0| +
                \frac{7}{8}|1 \rangle \langle 1|.

   Calculating the fidelity between the states :math:`\rho` and :math:`\sigma` as :math:`F(\rho, \sigma) \approx
   0.774`. This can be observed in :code:`|toqito⟩` as

   >>> from toqito.states import basis
   >>> from toqito.state_metrics import fidelity
   >>> e_0, e_1 = basis(2, 0), basis(2, 1)
   >>> rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
   >>> sigma = 1/8 * e_0 @ e_0.conj().T + 7/8 * e_1 @ e_1.conj().T
   >>> fidelity(rho, sigma)
   np.float64(0.77389339119464)

   As the sub-fidelity is a lower bound on the fidelity, that is :math:`E(\rho, \sigma) \leq F(\rho, \sigma)`, we can
   use :code:`|toqito⟩` to observe that :math:`E(\rho, \sigma) \approx 0.599\leq F(\rho, \sigma \approx 0.774`.

   >>> from toqito.states import basis
   >>> from toqito.state_metrics import sub_fidelity
   >>> e_0, e_1 = basis(2, 0), basis(2, 1)
   >>> rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
   >>> sigma = 1/8 * e_0 @ e_0.conj().T + 7/8 * e_1 @ e_1.conj().T
   >>> sub_fidelity(rho, sigma)
   np.float64(0.5989109809347399)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices are not of equal dimension.
   :param rho: Density operator.
   :param sigma: Density operator.
   :return: The sub-fidelity between :code:`rho` and :code:`sigma`.




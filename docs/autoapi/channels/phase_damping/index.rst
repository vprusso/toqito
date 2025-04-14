channels.phase_damping
======================

.. py:module:: channels.phase_damping

.. autoapi-nested-parse::

   phase damping channel.



Functions
---------

.. autoapisummary::

   channels.phase_damping.phase_damping


Module Contents
---------------

.. py:function:: phase_damping(input_mat = None, gamma = 0)

   Apply the phase damping channel to a quantum state :cite:`Chuang_2011_Quantum`.

   The phase damping channel describes how quantum information is lost due to environmental interactions,
   causing dephasing in the computational basis without losing energy.

   The Kraus operators for the phase damping channel are:

   .. math::
       K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
       K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix},

   .. rubric:: Examples

   Applying the phase damping channel to a qubit state:

   >>> import numpy as np
   >>> from toqito.channels.phase_damping import phase_damping
   >>> rho = np.array([[1, 0.5], [0.5, 1]])
   >>> result = phase_damping(rho, gamma=0.2)
   >>> print(result)
   [[1.       +0.j 0.4472136+0.j]
    [0.4472136+0.j 1.       +0.j]]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param input_mat: The input matrix to apply the channel to.
                     If `None`, the function returns the Kraus operators.
   :param gamma: The dephasing rate (between 0 and 1), representing the probability of phase decoherence.
   :return: The transformed quantum state after applying the phase damping channel.
            If `input_mat` is `None`, returns the list of Kraus operators.



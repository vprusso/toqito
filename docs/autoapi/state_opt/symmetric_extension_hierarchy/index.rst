state_opt.symmetric_extension_hierarchy
=======================================

.. py:module:: state_opt.symmetric_extension_hierarchy

.. autoapi-nested-parse::

   Calculates the optimal value of symmetric extension hierarchy SDP (semi definite programs).



Functions
---------

.. autoapisummary::

   state_opt.symmetric_extension_hierarchy.symmetric_extension_hierarchy


Module Contents
---------------

.. py:function:: symmetric_extension_hierarchy(states, probs = None, level = 2, dim = None)

   Compute optimal value of the symmetric extension hierarchy SDP :cite:`Navascues_2008_Pure`.

   The probability of distinguishing a given set of states via PPT measurements serves as a natural
   upper bound to the value of obtaining via separable measurements. Due to the nature of separable
   measurements, it is not possible to optimize directly over these objects via semidefinite
   programming techniques.

   We can, however, construct a hierarchy of semidefinite programs that attains closer and closer
   approximations at the separable value via the techniques described in :cite:`Navascues_2008_Pure`.

   The mathematical form of this hierarchy implemented here is explicitly given from equation 4.55
   in :cite:`Cosentino_2015_QuantumState`.

   .. math::

       \begin{equation}
           \begin{aligned}
               \text{maximize:} \quad & \sum_{k=1}^N p_k \langle \rho_k, \mu(k) \rangle, \\
               \text{subject to:} \quad & \sum_{k=1}^N \mu(k) =
                                          \mathbb{I}_{\mathcal{X} \otimes \mathcal{Y}}, \\
                                       & \text{Tr}_{\mathcal{Y}_2 \otimes \ldots \otimes
                                         \mathcal{Y}_s}(X_k) = \mu(k), \\
                                       & \left( \mathbb{I}_{\mathcal{X}} \otimes
                                         \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ovee \ldots \ovee
                                         \mathcal{Y}_s} \right) X_k
                                         \left(\mathbb{I}_{\mathcal{X}} \otimes
                                         \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ovee \ldots \ovee
                                         \mathcal{Y}_s} \right)
                                         = X_k \\
                                       & \text{T}_{\mathcal{X}}(X_k) \in \text{Pos}\left(
                                           \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                           \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                       & \text{T}_{\mathcal{Y}_2 \otimes \ldots \otimes
                                           \mathcal{Y}_s}(X_k) \in \text{Pos}\left(
                                           \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                           \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                       & X_1, \ldots, X_N \in
                                         \text{Pos}\left(\mathcal{X} \otimes \mathcal{Y} \otimes
                                         \mathcal{Y}_2 \otimes \ldots \otimes \mathcal{Y}_s
                                         \right).
           \end{aligned}
       \end{equation}

   .. rubric:: Examples

   It is known from :cite:`Cosentino_2015_QuantumState` that distinguishing three Bell states along with a resource
   state :math:`|\tau_{\epsilon}\rangle` via separable measurements has the following closed form

   .. math::
       \frac{1}{3} \left(2 + \sqrt{1 - \epsilon^2} \right)

   where the resource state is defined as

   .. math::
       |\tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} |00\rangle +
                                  \sqrt{\frac{1-\epsilon}{2}} |11\rangle.

   The value of optimally distinguishing these states via PPT measurements is strictly larger than
   the value one obtains from separable measurements. Calculating the first level of the hierarchy
   provides for us the optimal value of PPT measurements.

   Consider a fixed value of :math:`\epsilon = 0.5`.

   >>> from toqito.states import basis, bell
   >>> from toqito.perms import swap
   >>> import numpy as np
   >>> from toqito.state_opt import symmetric_extension_hierarchy
   >>> e_0, e_1 = basis(2, 0), basis(2, 1)
   >>> e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)
   >>>
   >>> # Define the resource state.
   >>> eps = 0.5
   >>> eps_state = np.sqrt((1+eps)/2) * e_00 + np.sqrt((1-eps)/2) * e_11
   >>> eps_dm = eps_state @ eps_state.conj().T
   >>>
   >>> # Define the ensemble of states to be distinguished.
   >>> states = [
   ...     np.kron(bell(0) @ bell(0).conj().T, eps_dm),
   ...     np.kron(bell(1) @ bell(1).conj().T, eps_dm),
   ...     np.kron(bell(2) @ bell(2).conj().T, eps_dm),
   ...     np.kron(bell(3) @ bell(3).conj().T, eps_dm),
   ... ]
   >>>
   >>> # Ensure the distinguishability is conducted on the proper spaces.
   >>> states = [
   ...     swap(states[0], [2, 3], [2, 2, 2, 2]),
   ...     swap(states[1], [2, 3], [2, 2, 2, 2]),
   ...     swap(states[2], [2, 3], [2, 2, 2, 2]),
   ... ]
   >>>
   >>> # Calculate the first level of the symmetric extension hierarchy. This
   >>> # is simply the value of optimally distinguishing via PPT measurements.
   >>> # np.around(symmetric_extension_hierarchy(states=states, probs=None, level=1), decimals=2)
   # 0.99
   >>>
   >>> # Calculating the second value gets closer to the separable value.
   >>> # np.around(symmetric_extension_hierarchy(states=states, probs=None, level=2), decimals=2)
   # 0.96
   >>>
   >>> # As proven in :cite:`Cosentino_2015_QuantumState`, the true separable value of distinguishing the
   >>> # three Bell states is:
   >>> # np.around(1/3 * (2 + np.sqrt(1 - eps**2)), decimals=2)
   # 0.96
   >>>
   >>> # Computing further levels of the hierarchy would eventually converge to
   >>> # this value, however, the higher the level, the more computationally
   >>> # demanding the SDP becomes.

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param states: A list of states provided as either matrices or vectors.
   :param probs: Respective list of probabilities each state is selected.
   :param level: Level of the hierarchy to compute.
   :param dim: The default has both subsystems of equal dimension.
   :return: The optimal probability of the symmetric extension hierarchy SDP for level
           :code:`level`.




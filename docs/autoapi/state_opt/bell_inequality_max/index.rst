state_opt.bell_inequality_max
=============================

.. py:module:: state_opt.bell_inequality_max

.. autoapi-nested-parse::

   Computes the upper bound for a given bipartite Bell inequality.



Functions
---------

.. autoapisummary::

   state_opt.bell_inequality_max.bell_inequality_max


Module Contents
---------------

.. py:function:: bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val, solver_name = 'SCS')

   Return the upper bound for the maximum violation(Tsirelson Bound) for a given bipartite Bell inequality.

   This computes the upper bound for the maximum value of a given bipartite Bell inequality using an SDP.
   The method is from :cite:`Navascues_2014_Characterization` and the implementation is based on :cite:`QETLAB_link`.
   This is useful for various tasks in device independent quantum information processing.

   The function formulates the problem as a SDP problem in the following format for the :math:`W`-state.

   .. math::

       \begin{multline}
       \max \operatorname{tr}\!\Bigl( W \cdot \sum_{a,b,x,y} B^{xy}_{ab}\, M^x_a \otimes N^y_b \Bigr),\\[1ex]
       \text{s.t.} \quad \operatorname{tr}(W) = 1,\quad W \ge 0,\\[1ex]
       W^{T_P} \ge 0,\quad \text{for all bipartitions } P.
       \end{multline}


   .. rubric:: Examples

   Consider the I3322 Bell inequality from :cite:`Collins_2004`.

   .. math::

       \begin{aligned}
       I_{3322} &= P(A_1 = B_1) + P(B_1 = A_2) + P(A_2 = B_2) + P(B_2 = A_3) \\
                &\quad - P(A_1 = B_2) - P(A_2 = B_3) - P(A_3 = B_1) - P(A_3 = B_3) \\
                &\le 2
       \end{aligned}

   The individual and joint coefficents and measurement values are encoded as matrices.
   The upper bound can then be found in :code:`|toqito⟩` as follows.

   >>> from toqito.state_opt import bell_inequality_max
   >>> import numpy as np
   >>> joint_coe = np.array([
   ... [1, 1, -1],
   ... [1, 1, 1],
   ... [-1, 1, 0]
   ... ])
   >>> a_coe = np.array([0, -1, 0])
   >>> b_coe = np.array([-1, -2, 0])
   >>> a_val = np.array([0, 1])
   >>> b_val = np.array([0, 1])
   >>> '%.3f' % bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
   '0.250'

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If `a_val` or `b_val` are not length 2.
   :param joint_coe: The coefficents for terms containing both A and B.
   :param a_coe: The coefficent for terms only containing A.
   :param b_coe: The coefficent for terms only containing B.
   :param a_val: The value of each measurement outcome for A.
   :param b_val: The value of each measurement outcome for B.
   :param solver_name: The solver used.
   :return: The upper bound for the maximum violation of the Bell inequality.



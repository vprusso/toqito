state_props.has_symmetric_extension
===================================

.. py:module:: state_props.has_symmetric_extension

.. autoapi-nested-parse::

   Determine whether there exists a symmetric extension for a given quantum state.



Functions
---------

.. autoapisummary::

   state_props.has_symmetric_extension.has_symmetric_extension


Module Contents
---------------

.. py:function:: has_symmetric_extension(rho, level = 2, dim = None, ppt = True, tol = 0.0001)

   Determine whether there exists a symmetric extension for a given quantum state.

   For more information, see :cite:`Doherty_2002_Distinguishing`.

   Determining whether an operator possesses a symmetric extension at some level :code:`level`
   can be used as a check to determine if the operator is entangled or not.

   This function was adapted from QETLAB.

   .. rubric:: Examples

   2-qubit symmetric extension:

   In :cite:`Chen_2014_Symmetric`, it was shown that a 2-qubit state :math:`\rho_{AB}` has a
   symmetric extension if and only if

   .. math::
       \text{Tr}(\rho_B^2) \geq \text{Tr}(\rho_{AB}^2) - 4 \sqrt{\text{det}(\rho_{AB})}.

   This closed-form equation is much quicker to check than running the semidefinite program.

   >>> import numpy as np
   >>> from toqito.state_props import has_symmetric_extension
   >>> from toqito.channels import partial_trace
   >>> rho = np.array([[1, 0, 0, -1],
   ...                 [0, 1, 1/2, 0],
   ...                 [0, 1/2, 1, 0],
   ...                 [-1, 0, 0, 1]])
   >>> # Show the closed-form equation holds
   >>> np.trace(np.linalg.matrix_power(partial_trace(rho, 1), 2)) >= np.trace(rho**2) - 4 * np.sqrt(np.linalg.det(rho))
   np.True_
   >>> # Now show that the `has_symmetric_extension` function recognizes this case.
   >>> has_symmetric_extension(rho)
   True

   Higher qubit systems:

   Consider a density operator corresponding to one of the Bell states.

   .. math::
       \rho = \frac{1}{2} \begin{pmatrix}
                           1 & 0 & 0 & 1 \\
                           0 & 0 & 0 & 0 \\
                           0 & 0 & 0 & 0 \\
                           1 & 0 & 0 & 1
                          \end{pmatrix}

   To make this state over more than just two qubits, let's construct the following state

   .. math::
       \sigma = \rho \otimes \rho.

   As the state :math:`\sigma` is entangled, there should not exist a symmetric extension at some
   level. We see this being the case for a relatively low level of the hierarchy.

   >>> import numpy as np
   >>> from toqito.states import bell
   >>> from toqito.state_props import has_symmetric_extension
   >>>
   >>> rho = bell(0) @ bell(0).conj().T
   >>> sigma = np.kron(rho, rho)
   >>> has_symmetric_extension(sigma)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If dimension does not evenly divide matrix length.
   :param rho: A matrix or vector.
   :param level: Level of the hierarchy to compute.
   :param dim: The default has both subsystems of equal dimension.
   :param ppt: If :code:`True`, this enforces that the symmetric extension must be PPT.
   :param tol: Tolerance when determining whether a symmetric extension exists.
   :return: :code:`True` if :code:`mat` has a symmetric extension; :code:`False` otherwise.




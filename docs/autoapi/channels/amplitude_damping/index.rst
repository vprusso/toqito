channels.amplitude_damping
==========================

.. py:module:: channels.amplitude_damping

.. autoapi-nested-parse::

   Generates the (generalized) amplitude damping channel.



Functions
---------

.. autoapisummary::

   channels.amplitude_damping.amplitude_damping


Module Contents
---------------

.. py:function:: amplitude_damping(input_mat = None, gamma = 0, prob = 1)

   Apply the generalized amplitude damping channel to a quantum state.

   The generalized amplitude damping channel is a quantum channel that models energy dissipation
   in a quantum system, where the system can lose energy to its environment with a certain
   probability. This channel is defined by two parameters: `gamma` (the damping rate) and `prob`
   (the probability of energy loss).

   To also include standard implementation of amplitude damping, we have set `prob = 1` as the default implementation.

   .. note::
         This channel is defined for qubit systems in the standard literature :cite:`Khatri_2020_Information`.


   The Kraus operators for the generalized amplitude damping channel are given by:

   .. math::
       K_0 = \sqrt{p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
       K_1 = \sqrt{p}  \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}, \\
       K_2 = \sqrt{1 - p} \begin{pmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{pmatrix}, \\
       K_3 = \sqrt{1 - p}  \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}, \\

   These operators describe the evolution of a quantum state under the generalized amplitude
   damping process.

   .. rubric:: Examples

   Apply the generalized amplitude damping channel to a qubit state:

   >>> import numpy as np
   >>> from toqito.channels import amplitude_damping
   >>> rho = np.array([[1, 0], [0, 0]])  # |0><0|
   >>> result = amplitude_damping(rho, gamma=0.1, prob=0.5)
   >>> print(result)
   [[0.95+0.j 0.  +0.j]
    [0.  +0.j 0.05+0.j]]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param input_mat: The input matrix to which the channel is applied.
                     If `None`, the function returns the Kraus operators of the channel.
   :param gamma: The damping rate, a float between 0 and 1. Represents the probability of
                 energy dissipation.
   :param prob: The probability of energy loss, a float between 0 and 1.
   :return: The evolved quantum state after applying the generalized amplitude damping channel.
            If `input_mat` is `None`, it returns the list of Kraus operators.



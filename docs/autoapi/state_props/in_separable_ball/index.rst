state_props.in_separable_ball
=============================

.. py:module:: state_props.in_separable_ball

.. autoapi-nested-parse::

   Checks whether operator is in the ball of separability centered at the maximally-mixed state.



Functions
---------

.. autoapisummary::

   state_props.in_separable_ball.in_separable_ball


Module Contents
---------------

.. py:function:: in_separable_ball(mat)

   Check whether an operator is contained in ball of separability :cite:`Gurvits_2002_Largest`.

   Determines whether :code:`mat` is contained within the ball of separable operators centered
   at the identity matrix (i.e. the maximally-mixed state). The size of this ball was derived in
   :cite:`Gurvits_2002_Largest`.

   This function can be used as a method for separability testing of states in certain scenarios.

   This function is adapted from QETLAB.

   .. rubric:: Examples

   The only states acting on :math:`\mathbb{C}^m \otimes \mathbb{C}^n` in the
   separable ball that do not have full rank are those with exactly 1 zero
   eigenvalue, and the :math:`mn - 1` non-zero eigenvalues equal to each
   other.

   The following is an example of generating a random density matrix with eigenvalues
   :code:`[1, 1, 1, 0]/3`. This example yields a matrix that is contained within the separable
   ball.

   >>> from toqito.rand import random_unitary
   >>> from toqito.state_props import in_separable_ball
   >>> import numpy as np
   >>>
   >>> U = random_unitary(4)
   >>> lam = np.array([1, 1, 1, 0]) / 3
   >>> rho = U @ np.diag(lam) @ U.conj().T
   >>> in_separable_ball(rho)
   np.True_

   The following is an example of generating a random density matrix with eigenvalues
   :code:`[1.01, 1, 0.99, 0]/3`. This example yields a matrix that is not contained within the
   separable ball.

   >>> from toqito.rand import random_unitary
   >>> from toqito.state_props import in_separable_ball
   >>> import numpy as np
   >>>
   >>> U = random_unitary(4)
   >>> lam = np.array([1.01, 1, 0.99, 0]) / 3
   >>> rho = U @ np.diag(lam) @ U.conj().T
   >>> in_separable_ball(rho)
   np.False_

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: A positive semidefinite matrix or a vector of the eigenvalues of a positive
               semidefinite matrix.
   :return: :code:`True` if the matrix :code:`mat` is contained within the separable ball, and
           :code:`False` otherwise.




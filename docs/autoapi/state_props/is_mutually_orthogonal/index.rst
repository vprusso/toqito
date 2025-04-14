state_props.is_mutually_orthogonal
==================================

.. py:module:: state_props.is_mutually_orthogonal

.. autoapi-nested-parse::

   Checks if quantum states are mutually orthogonal.



Functions
---------

.. autoapisummary::

   state_props.is_mutually_orthogonal.is_mutually_orthogonal


Module Contents
---------------

.. py:function:: is_mutually_orthogonal(vec_list)

   Check if list of vectors are mutually orthogonal :cite:`WikiOrthog`.

   We say that two bases

   .. math::
       \begin{equation}
           \mathcal{B}_0 = \left\{u_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
           \quad \text{and} \quad
           \mathcal{B}_1 = \left\{v_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
       \end{equation}

   are *mutually orthogonal* if and only if
   :math:`\left|\langle u_a, v_b \rangle\right| = 0` for all :math:`a, b \in \Sigma`.

   For :math:`n \in \mathbb{N}`, a set of bases :math:`\left\{
   \mathcal{B}_0, \ldots, \mathcal{B}_{n-1} \right\}` are mutually orthogonal if and only if
   every basis is orthogonal with every other basis in the set, i.e. :math:`\mathcal{B}_x`
   is orthogonal with :math:`\mathcal{B}_x^{\prime}` for all :math:`x \not= x^{\prime}` with
   :math:`x, x^{\prime} \in \Sigma`.

   .. rubric:: Examples

   The Bell states constitute a set of mutually orthogonal vectors.

   >>> from toqito.states import bell
   >>> from toqito.state_props import is_mutually_orthogonal
   >>> states = [bell(0), bell(1), bell(2), bell(3)]
   >>> is_mutually_orthogonal(states)
   True

   The following is an example of a list of vectors that are not mutually orthogonal.

   >>> import numpy as np
   >>> from toqito.states import bell
   >>> from toqito.state_props import is_mutually_orthogonal
   >>> states = [np.array([1, 0]), np.array([1, 1])]
   >>> is_mutually_orthogonal(states)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If at least two vectors are not provided.
   :param vec_list: The list of vectors to check.
   :return: :code:`True` if :code:`vec_list` are mutually orthogonal, and
            :code:`False` otherwise.




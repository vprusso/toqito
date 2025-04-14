state_props.is_distinguishable
==============================

.. py:module:: state_props.is_distinguishable

.. autoapi-nested-parse::

   Checks if a set of quantum states are distinguishable.



Functions
---------

.. autoapisummary::

   state_props.is_distinguishable.is_distinguishable


Module Contents
---------------

.. py:function:: is_distinguishable(states, probs = None)

   Check whether a collection of vectors are (perfectly) distinguishable or not.

   The ability to determine whether a set of quantum states are distinguishable can be obtained via the state
   distinguishability SDP as defined in `state_distinguishability`

   .. rubric:: Examples

   The set of Bell states are an example of distinguishable states. Recall that the Bell states are defined as:

   .. math::
       u_1 = \frac{1}{\sqrt{2}} \left(|00\rangle + |11\rangle\right), &\quad
       u_2 = \frac{1}{\sqrt{2}} \left(|00\rangle - |11\rangle\right), \\
       u_3 = \frac{1}{\sqrt{2}} \left(|01\rangle + |10\rangle\right), &\quad
       u_4 = \frac{1}{\sqrt{2}} \left(|01\rangle - |10\rangle\right).

   It can be checked in :code`toqito` that the Bell states are distinguishable:

   >>> from toqito.states import bell
   >>> from toqito.state_props import is_distinguishable
   >>>
   >>> bell_states = [bell(0), bell(1), bell(2), bell(3)]
   >>> is_distinguishable(bell_states)
   np.True_

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param states: A set of vectors consisting of quantum states to determine the distinguishability of.
   :param probs: Respective list of probabilities each state is selected. If no
               probabilities are provided, a uniform probability distribution is assumed.
   :return: :code:`True` if the vectors are distinguishable; :code:`False` otherwise.




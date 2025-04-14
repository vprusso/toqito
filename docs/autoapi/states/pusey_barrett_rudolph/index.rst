states.pusey_barrett_rudolph
============================

.. py:module:: states.pusey_barrett_rudolph

.. autoapi-nested-parse::

   Construct a set of mutually unbiased bases.



Functions
---------

.. autoapisummary::

   states.pusey_barrett_rudolph.pusey_barrett_rudolph


Module Contents
---------------

.. py:function:: pusey_barrett_rudolph(n, theta)

   Produce set of Pusey-Barrett-Rudolph (PBR) states :cite:`Pusey_2012_On`.

   Let :math:`\theta \in [0, \pi/2]` be an angle. Define the states

   .. math::
       |\psi_0\rangle = \cos(\frac{\theta}{2})|0\rangle +
                        \sin(\frac{\theta}{2})|1\rangle
       \quad \text{and} \quad
       |\psi_1\rangle = \cos(\frac{\theta}{2})|0\rangle -
                        \sin(\frac{\theta}{2})|1\rangle.

   For some :math:`n \geq 1`, define a basis of :math:`2^n` states where

   .. math::
       |\Psi_i\rangle = |\psi_{x_i}\rangle \otimes \cdots \otimes |\psi_{x_n}\rangle.

   These PBR states are defined in Equation (A6) from :cite:`Pusey_2012_On`.

   .. rubric:: Examples

   Generating the PBR states can be done by simply invoking the function with a given choice of :code:`n` and
   :code:`theta`:

   >>> from toqito.states import pusey_barrett_rudolph
   >>> pusey_barrett_rudolph(n=1, theta=0.5)
   [array([[0.96891242],
   ...    [0.24740396]]), array([[ 0.96891242],
   ...    [-0.24740396]])]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param n: The number of states in the set.
   :param theta: Angle parameter that defines the states.
   :return: Vector of trine states.



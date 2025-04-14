states.brauer
=============

.. py:module:: states.brauer

.. autoapi-nested-parse::

   Brauer states are the p_val-fold tensor product of the standard maximally-entangled pure states.



Functions
---------

.. autoapisummary::

   states.brauer.brauer


Module Contents
---------------

.. py:function:: brauer(dim, p_val)

   Produce all Brauer states :cite:`WikiBrauer`.

   Produce a matrix whose columns are all of the (unnormalized) "Brauer" states: states that are the :code:`p_val`-fold
   tensor product of the standard maximally-entangled pure state on :code:`dim` local dimensions. There are many such
   states, since there are many different ways to group the :code:`2 * p_val` parties into :code:`p_val` pairs (with
   each pair corresponding to one maximally-entangled state).

   The exact number of such states is:

   >>> import math
   >>> import numpy as np
   >>> p_val = 2
   >>> math.factorial(2 * p_val) / (math.factorial(p_val) * 2**p_val)
   3.0

   which is the number of columns of the returned matrix.

   This function has been adapted from QETLAB.

   .. rubric:: Examples

   Generate a matrix whose columns are all Brauer states on 4 qubits.

   >>> from toqito.states import brauer
   >>> brauer(2, 2)
   array([[1., 1., 1.],
          [0., 0., 0.],
          [0., 0., 0.],
          [1., 0., 0.],
          [0., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 1.],
          [0., 1., 0.],
          [0., 0., 0.],
          [1., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [1., 1., 1.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: Dimension of each local subsystem
   :param p_val: Half of the number of parties (i.e., the state that this function computes will
                 live in :math:`(\mathbb{C}^D)^{\otimes 2 P})`
   :return: Matrix whose columns are all of the unnormalized Brauer states.




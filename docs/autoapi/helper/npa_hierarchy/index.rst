helper.npa_hierarchy
====================

.. py:module:: helper.npa_hierarchy

.. autoapi-nested-parse::

   Generates the NPA constraints.



Classes
-------

.. autoapisummary::

   helper.npa_hierarchy.Symbol


Functions
---------

.. autoapisummary::

   helper.npa_hierarchy._reduce
   helper.npa_hierarchy._parse
   helper.npa_hierarchy._gen_words
   helper.npa_hierarchy._is_zero
   helper.npa_hierarchy._is_meas
   helper.npa_hierarchy._is_meas_on_one_player
   helper.npa_hierarchy._get_nonlocal_game_params
   helper.npa_hierarchy.npa_constraints


Module Contents
---------------

.. py:class:: Symbol

   Bases: :py:obj:`tuple`


   .. py:attribute:: player


   .. py:attribute:: question


   .. py:attribute:: answer


.. py:function:: _reduce(word)

.. py:function:: _parse(k)

.. py:function:: _gen_words(k, a_out, a_in, b_out, b_in)

.. py:function:: _is_zero(word)

.. py:function:: _is_meas(word)

.. py:function:: _is_meas_on_one_player(word)

.. py:function:: _get_nonlocal_game_params(assemblage, referee_dim = 1)

.. py:function:: npa_constraints(assemblage, k = 1, referee_dim = 1)

   Generate the constraints specified by the NPA hierarchy up to a finite level :cite:`Navascues_2008_AConvergent`.

   You can determine the level of the hierarchy by a positive integer or a string
   of a form like "1+ab+aab", which indicates that an intermediate level of the hierarchy
   should be used, where this example uses all products of 1 measurement, all products of
   one Alice and one Bob measurement, and all products of two Alice and one Bob measurement.

   The commuting measurement assemblage operator must be given as a dictionary. The keys are
   tuples of Alice and Bob questions :math:`x, y` and the values are cvxpy Variables which
   are matrices with entries:

   .. math::
       K_{xy}\Big(i + a \cdot dim_R, j + b \cdot dim_R \Big) =
       \langle i| \text{Tr}_{\mathcal{H}} \Big( \big(
           I_R \otimes A_a^x B_b^y \big) \sigma \Big) |j \rangle

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param assemblage: The commuting measurement assemblage operator.
   :param k: The level of the NPA hierarchy to use (default=1).
   :param referee_dim: The dimension of the referee's quantum system (default=1).
   :return: A list of cvxpy constraints.



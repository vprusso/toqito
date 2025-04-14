perms.perfect_matchings
=======================

.. py:module:: perms.perfect_matchings

.. autoapi-nested-parse::

   Perfect matchings refers to ways of grouping an even number of objects into pairs.



Functions
---------

.. autoapisummary::

   perms.perfect_matchings.perfect_matchings


Module Contents
---------------

.. py:function:: perfect_matchings(num)

   Give all perfect matchings of :code:`num` objects.

   The input can be either an even natural number (the number of objects to be matched) or a `numpy` array containing
   an even number of distinct objects to be matched.

   Returns all perfect matchings of a given list of objects. That is, it returns all ways of grouping an even number of
   objects into pairs.

   This function is adapted from QETLAB. :cite:`QETLAB_link`.

   .. rubric:: Examples

   This is an example of how to generate all perfect matchings of the numbers 0, 1, 2, 3.

   >>> from toqito.perms import perfect_matchings
   >>> perfect_matchings(4)
   array([[0, 1, 2, 3],
          [0, 2, 1, 3],
          [0, 3, 2, 1]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param num: Either an even integer, indicating that you would like all perfect matchings of the
               integers 0, 1, ... N-1, or a `list` or `np.array` containing an even number of distinct
               entries, indicating that you would like all perfect matchings of those entries.
   :return: An array containing all valid perfect matchings of size :code:`num`.



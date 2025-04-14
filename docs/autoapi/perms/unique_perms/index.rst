perms.unique_perms
==================

.. py:module:: perms.unique_perms

.. autoapi-nested-parse::

   Unique permutations is used to calculate the unique permutations of a list/vector and their count.



Classes
-------

.. autoapisummary::

   perms.unique_perms.UniqueElement


Functions
---------

.. autoapisummary::

   perms.unique_perms.unique_perms
   perms.unique_perms.perm_unique_helper


Module Contents
---------------

.. py:class:: UniqueElement

   Class for unique elements to keep track of occurrences.


   .. py:attribute:: value
      :type:  int


   .. py:attribute:: occurrences
      :type:  int


.. py:function:: unique_perms(elements)

   Determine the number of unique permutations of a list.

   .. rubric:: Examples

   Consider the following vector

   .. math::
       \left[1, 1, 2, 2, 1, 2, 1, 3, 3, 3\right].

   The number of possible permutations possible with the above vector is :math:`4200`. This can be
   obtained using the :code:`|toqito⟩` package as follows.

   >>> from toqito.perms import unique_perms
   >>> vec_nums = [1, 1, 2, 2, 1, 2, 1, 3, 3, 3]
   >>> len(list(unique_perms(vec_nums)))
   4200

   :param elements: List of integers.
   :return: The number of possible permutations possible.



.. py:function:: perm_unique_helper(list_unique, result_list, elem_d)

   Provide helper function for unique_perms.

   :param list_unique:
   :param result_list:
   :param elem_d:
   :return:



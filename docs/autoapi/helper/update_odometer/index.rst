helper.update_odometer
======================

.. py:module:: helper.update_odometer

.. autoapi-nested-parse::

   Updates the odometer.



Functions
---------

.. autoapisummary::

   helper.update_odometer.update_odometer


Module Contents
---------------

.. py:function:: update_odometer(old_ind, upper_lim)

   Increase a vector as odometer.

   Increases the last entry of the vector `old_ind` by 1, unless that would
   make it larger than the last entry of the vector `upper_lim`. In this case,
   it sets the last entry to 0 and instead increases the second-last entry of
   `old_ind`, unless that would make it larger than the second-last entry of
   `upper_lim`. In this case, it sets the second-last entry to 0 and instead
   increases the third-last entry of `old_ind` (and so on; it works like an
   odometer).

   This function is useful when you want to have k nested loops, but k isn't
   specified beforehand. For example, instead of looping over i and j going
   from 1 to 3, you could loop over a single variable going from 1 to 3^2 and
   set [i, j] = update_odometer([i, j], [3, 3]) at each step within the loop.

   This function is adapted from QETLAB :cite:`QETLAB_link`.

   .. rubric:: Examples

   >>> from toqito.helper import update_odometer
   >>> import numpy as np
   >>> vec = np.array([0, 0])
   >>> upper_lim = np.array([3, 2])
   >>> for j in range(0, np.prod(upper_lim)-1):
   ...     vec = update_odometer(vec, upper_lim)
   ...     vec
   array([0, 1])
   array([1, 0])
   array([1, 1])
   array([2, 0])
   array([2, 1])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param old_ind: The initial vector.
   :param upper_lim: The upper limit on which to increase the odometer to.
   :return: The updated vector.




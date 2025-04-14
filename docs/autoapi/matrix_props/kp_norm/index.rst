matrix_props.kp_norm
====================

.. py:module:: matrix_props.kp_norm

.. autoapi-nested-parse::

   Computes the Kp-norm for matrices or vectors.



Functions
---------

.. autoapisummary::

   matrix_props.kp_norm.kp_norm


Module Contents
---------------

.. py:function:: kp_norm(mat, k, p)

   Compute the kp_norm of vector or matrix.

   Calculate the p-norm of a vector or the k-largest singular values of a
   matrix.

   .. rubric:: Examples

   To compute the p-norm of a vector:

   >>> import numpy as np
   >>> from toqito.matrix_props import kp_norm
   >>> from toqito.states import bell
   >>> np.around(kp_norm(bell(0), 1, np.inf), decimals=2)
   np.float64(1.0)

   To compute the k-largest singular values of a matrix:

   >>> import numpy as np
   >>> from toqito.matrix_props import kp_norm
   >>> from toqito.rand import random_unitary
   >>> np.around(kp_norm(random_unitary(5), 5, 2), decimals=2)
   np.float64(2.24)

   :param mat: 2D numpy ndarray
   :param k: The number of singular values to take.
   :param p: The order of the norm.
   :return: The kp-norm of a matrix.




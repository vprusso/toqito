matrix_ops.perturb_vectors
==========================

.. py:module:: matrix_ops.perturb_vectors

.. autoapi-nested-parse::

   Perturb vectors is used to add a small random number to each element of a vector.

   A random value is added sampled from a normal distribution scaled by `eps`.



Functions
---------

.. autoapisummary::

   matrix_ops.perturb_vectors.perturb_vectors


Module Contents
---------------

.. py:function:: perturb_vectors(vectors, eps = 0.1)

   Perturb the vectors by adding a small random number to each element.

   :param vectors: List of vectors to perturb.
   :param eps: Amount by which to perturb vectors.
   :return: Resulting list of perturbed vectors by a factor of epsilon.

   Example:
   ==========

       >>> from toqito.matrix_ops import perturb_vectors
       >>> import numpy as np
       >>> vectors = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
       >>> perturb_vectors(vectors, eps=0.1) # doctest: +SKIP
       array([[0.47687587, 0.87897065],
              [0.58715549, 0.80947417]])




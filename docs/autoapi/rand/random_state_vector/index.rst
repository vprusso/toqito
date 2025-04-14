rand.random_state_vector
========================

.. py:module:: rand.random_state_vector

.. autoapi-nested-parse::

   Generates a random state vector.



Functions
---------

.. autoapisummary::

   rand.random_state_vector.random_state_vector


Module Contents
---------------

.. py:function:: random_state_vector(dim, is_real = False, k_param = 0, seed = None)

   Generate a random pure state vector.

   .. rubric:: Examples

   We may generate a random state vector. For instance, here is an example where we can generate a
   :math:`2`-dimensional random state vector.

   >>> from toqito.rand import random_state_vector
   >>> vec = random_state_vector(2)
   >>> vec # doctest: +SKIP
   array([[0.78645233+0.16239043j],
          [0.56649582+0.18494478j]])

   We can verify that this is in fact a valid state vector by computing the corresponding density matrix of the vector
   and checking if the density matrix is pure.

   >>> from toqito.state_props import is_pure
   >>> dm = vec.conj().T @ vec
   >>> is_pure(dm)
   True

   It is also possible to pass a seed for reproducibility.

   >>> from toqito.rand import random_state_vector
   >>> vec = random_state_vector(2, seed=42)
   >>> vec
   array([[0.54521054+0.60483621j],
          [0.30916633+0.49125839j]])

   We can once again verify that this is in fact a valid state vector by computing the
   corresponding density matrix of the vector and checking if the density matrix is pure.

   >>> from toqito.state_props import is_pure
   >>> dm = vec.conj().T @ vec
   >>> is_pure(dm)
   True



   :param dim: The number of rows (and columns) of the unitary matrix.
   :param is_real: Boolean denoting whether the returned matrix has real
                   entries or not. Default is :code:`False`.
   :param k_param: Default 0.
   :param seed: A seed used to instantiate numpy's random number generator.
   :return: A :code:`dim`-by-:code:`dim` random unitary matrix.




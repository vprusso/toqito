state_props.sk_vec_norm
=======================

.. py:module:: state_props.sk_vec_norm

.. autoapi-nested-parse::

   Compute the S(k)-norm of a vector.



Functions
---------

.. autoapisummary::

   state_props.sk_vec_norm.sk_vector_norm


Module Contents
---------------

.. py:function:: sk_vector_norm(rho, k = 1, dim = None)

   Compute the S(k)-norm of a vector :cite:`Johnston_2010_AFamily`.

   The :math:`S(k)`-norm of of a vector :math:`|v \rangle` is
   defined as:

   .. math::
       \big|\big| |v\rangle \big|\big|_{s(k)} := \text{sup}_{|w\rangle} \Big\{
           |\langle w | v \rangle| : \text{Schmidt-rank}(|w\rangle) \leq k
       \Big\}

   It's also equal to the Euclidean norm of the vector of :math:`|v\rangle`'s
   k largest Schmidt coefficients.

   This function was adapted from QETLAB.

   .. rubric:: Examples

   The smallest possible value of the :math:`S(k)`-norm of a pure state is
   :math:`\sqrt{\frac{k}{n}}`, and is attained exactly by the "maximally entangled
   states".

   >>> from toqito.states import max_entangled
   >>> from toqito.state_props import sk_vector_norm
   >>> import numpy as np
   >>>
   >>> # Maximally entagled state.
   >>> v = max_entangled(4)
   >>> sk_vector_norm(v)
   np.float64(0.5)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param rho: A vector.
   :param k: An int.
   :param dim: The dimension of the two sub-systems. By default it's
               assumed to be equal.
   :return: The S(k)-norm of :code:`rho`.




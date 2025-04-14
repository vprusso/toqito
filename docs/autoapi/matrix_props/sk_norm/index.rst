matrix_props.sk_norm
====================

.. py:module:: matrix_props.sk_norm

.. autoapi-nested-parse::

   Computes the S(k)-norm of a matrix.



Functions
---------

.. autoapisummary::

   matrix_props.sk_norm.sk_operator_norm
   matrix_props.sk_norm.__target_is_proved
   matrix_props.sk_norm.__lower_bound_sk_norm_randomized


Module Contents
---------------

.. py:function:: sk_operator_norm(mat, k = 1, dim = None, target = None, effort = 2)

   Compute the S(k)-norm of a matrix :cite:`Johnston_2010_AFamily`.

   The :math:`S(k)`-norm of of a matrix :math:`X` is defined as:

   .. math::
       \big|\big| X \big|\big|_{S(k)} := sup_{|v\rangle, |w\rangle}
       \Big\{
           |\langle w | X |v \rangle| :
           \text{Schmidt - rank}(|v\rangle) \leq k,
           \text{Schmidt - rank}(|w\rangle) \leq k
       \Big\}

   Since computing the exact value of S(k)-norm :cite:`Johnston_2012_Norms` is in the general case an intractable
   problem, this function tries to find some good lower and upper bounds. You can control the amount of computation you
   want to devote to computing the bounds by `effort` input argument. Note that if the input matrix is not positive
   semidefinite the output bounds might be quite poor.

   This function was adapted from QETLAB.

   .. rubric:: Examples

   The :math:`S(1)`-norm of a Werner state :math:`\rho_a \in M_n \otimes M_n` is

   .. math::
       \big|\big| \rho_a \big|\big|_{S(1)} = \frac{1 + |min\{a, 0\}|}{n (n - a)}

   >>> from toqito.states.werner import werner
   >>> from toqito.matrix_props.sk_norm import sk_operator_norm
   >>> # Werner state.
   >>> n = 4; a = 0
   >>> rho = werner(4, 0.)
   >>> sk_operator_norm(rho)
   (np.float64(0.0625), np.float64(0.0625))

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If dimension of the input matrix is not specified.
   :param mat: A matrix.
   :param k: The "index" of the norm--that is, it is the Schmidt rank of the
             vectors that are multiplying X on the left and right in the definition
             of the norm.
   :param dim: The dimension of the two sub-systems. By default it's
               assumed to be equal.
   :param target: A target value that you wish to prove that the norm is above or below.
   :param effort: An integer value indicating the amount of computation you want to
                  devote to computing the bounds.
   :return: A lower and an upper bound on S(k)-norm of :code:`mat`.



.. py:function:: __target_is_proved(lower_bound, upper_bound, op_norm, tol, target)

.. py:function:: __lower_bound_sk_norm_randomized(mat, k = 1, dim = None, tol = 1e-05, start_vec = None)


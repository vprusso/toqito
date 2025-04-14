helper.channel_dim
==================

.. py:module:: helper.channel_dim

.. autoapi-nested-parse::

   Channel dimensions coputes and returns the input, output and environment dimensions of a channel.



Functions
---------

.. autoapisummary::

   helper.channel_dim.channel_dim
   helper.channel_dim._expand_dim


Module Contents
---------------

.. py:function:: channel_dim(phi, allow_rect = True, dim = None, compute_env_dim = True)

   Compute the input, output, and environment dimensions of a channel.

   This function returns the dimensions of the input, output, and environment spaces of
   input channel, in that order. Input and output dimensions are both 1-by-2 vectors
   containing the row and column dimensions of their spaces. The enviroment dimension
   is always a scalar, and it is equal to the number of Kraus operators of PHI (if PHI is
   provided as a Choi matrix then enviroment dimension is the *minimal* number of Kraus
   operators of any representation of PHI).

   Input DIM should provided if and only if PHI is a Choi matrix with unequal input and
   output dimensions (since it is impossible to determine the input and output dimensions
   from the Choi matrix alone). If ALLOW_RECT is false and PHI acts on non-square matrix
   spaces, an error will be produced. If PHI maps M_{r,c} to M_{x,y} then DIM should be the
   2-by-2 matrix [[r,x], [c,y]]. If PHI maps M_m to M_n, then DIM can simply be the vector
   [m,n]. If ALLOW_RECT is false then returned input and output dimensions will be scalars
   instead of vectors. If COMPUTE_ENV_DIM is false and the PHI is a Choi matrix we avoid
   computing the rank of the Choi matrix.

   This functions was adapted from QETLAB :cite:`QETLAB_link`.

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: A superoperator. It should be provided either as a Choi matrix,
               or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
   :param allow_rect: A flag indicating that the input and output spaces of PHI can be non-square (default True).
   :param dim: A scalar, vector or matrix containing the input and output dimensions of PHI.
   :param compute_env_dim: A flag indicating whether we compute the enviroment dimension.
   :return: The input, output, and environment dimensions of a channel.


.. py:function:: _expand_dim(dim)


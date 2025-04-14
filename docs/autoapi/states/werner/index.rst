states.werner
=============

.. py:module:: states.werner

.. autoapi-nested-parse::

   Werner states.

   Werner states are mixtures of projectors onto the symmetric and permutation operator that exchanges the two subsystems.



Functions
---------

.. autoapisummary::

   states.werner.werner


Module Contents
---------------

.. py:function:: werner(dim, alpha)

   Produce a Werner state :cite:`Werner_1989_QuantumStates`.

   A Werner state is a state of the following form

   .. math::

       \begin{equation}
           \rho_{\alpha} = \frac{1}{d^2 - d\alpha} \left(\mathbb{I} \otimes
           \mathbb{I} - \alpha S \right) \in \mathbb{C}^d \otimes \mathbb{C}^d.
       \end{equation}

   Yields a Werner state with parameter :code:`alpha` acting on :code:`(dim * dim)`- dimensional space. More
   specifically, :math:`\rho` is the density operator defined by :math:`(\mathbb{I} - `alpha` S)` (normalized to have
   trace 1), where :math:`\mathbb{I}` is the density operator and :math:`S` is the operator that swaps two copies of
   :code:`dim`-dimensional space (see swap and swap_operator for example).

   If :code:`alpha` is a vector with :math:`p!-1` entries, for some integer :math:`p > 1`, then a multipartite Werner
   state is returned. This multipartite Werner state is the normalization of I - `alpha(1)*P(2)` - ... -
   `alpha(p!-1)*P(p!)`, where P(i) is the operator that permutes p subsystems according to the i-th permutation when
   they are written in lexicographical order (for example, the lexicographical ordering when p = 3 is: `[1, 2, 3], [1,
   3, 2], [2, 1,3], [2, 3, 1], [3, 1, 2], [3, 2, 1],` so P(4) in this case equals permutation_operator(dim, [2, 3, 1]).

   .. rubric:: Examples

   Computing the qutrit Werner state with :math:`\alpha = 1/2` can be done in :code:`|toqito⟩` as

   >>> from toqito.states import werner
   >>> werner(3, 1 / 2)
   array([[ 0.06666667,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.13333333,  0.        , -0.06666667,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.13333333,  0.        ,  0.        ,
            0.        , -0.06666667,  0.        ,  0.        ],
          [ 0.        , -0.06666667,  0.        ,  0.13333333,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.06666667,
            0.        ,  0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.13333333,  0.        , -0.06666667,  0.        ],
          [ 0.        ,  0.        , -0.06666667,  0.        ,  0.        ,
            0.        ,  0.13333333,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           -0.06666667,  0.        ,  0.13333333,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.06666667]])

   We may also compute multipartite Werner states in :code:`|toqito⟩` as well.

   >>> from toqito.states import werner
   >>> werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
   array([[ 0.11286089,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.12729659, -0.00787402,  0.        , -0.00656168,
            0.        ,  0.        ,  0.        ],
          [ 0.        , -0.00918635,  0.1312336 ,  0.        , -0.00918635,
            0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.12860892,  0.        ,
           -0.01049869, -0.00524934,  0.        ],
          [ 0.        , -0.00524934, -0.01049869,  0.        ,  0.12860892,
            0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        , -0.00918635,  0.        ,
            0.1312336 , -0.00918635,  0.        ],
          [ 0.        ,  0.        ,  0.        , -0.00656168,  0.        ,
           -0.00787402,  0.12729659,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.11286089]])


   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: Alpha vector does not have the correct length.
   :param dim: The dimension of the Werner state.
   :param alpha: Parameter to specify Werner state.
   :return: A Werner state of dimension :code:`dim`.



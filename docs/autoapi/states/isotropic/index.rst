states.isotropic
================

.. py:module:: states.isotropic

.. autoapi-nested-parse::

   Isotropic state is a bipartite quantum state.

   These states are separable for α ≤ 1/(d+1), but are otherwise entangled.



Functions
---------

.. autoapisummary::

   states.isotropic.isotropic


Module Contents
---------------

.. py:function:: isotropic(dim, alpha)

   Produce a isotropic state :cite:`Horodecki_1998_Reduction`.

   Returns the isotropic state with parameter :code:`alpha` acting on (:code:`dim`-by-:code:`dim`)-dimensional space.
   The isotropic state has the following form

   .. math::
       \begin{equation}
           \rho_{\alpha} = \frac{1 - \alpha}{d^2} \mathbb{I} \otimes
           \mathbb{I} + \alpha |\psi_+ \rangle \langle \psi_+ | \in
           \mathbb{C}^d \otimes \mathbb{C}^2
       \end{equation}

   where :math:`|\psi_+ \rangle = \frac{1}{\sqrt{d}} \sum_j |j \rangle \otimes |j \rangle` is the maximally entangled
   state.

   .. rubric:: Examples

   To generate the isotropic state with parameter :math:`\alpha=1/2`, we can make the following call to
   :code:`|toqito⟩` as

   >>> from toqito.states import isotropic
   >>> isotropic(3, 1 / 2)
   array([[0.22222222, 0.        , 0.        , 0.        , 0.16666667,
           0.        , 0.        , 0.        , 0.16666667],
          [0.        , 0.05555556, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.05555556, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.05555556, 0.        ,
           0.        , 0.        , 0.        , 0.        ],
          [0.16666667, 0.        , 0.        , 0.        , 0.22222222,
           0.        , 0.        , 0.        , 0.16666667],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05555556, 0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.05555556, 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.05555556, 0.        ],
          [0.16666667, 0.        , 0.        , 0.        , 0.16666667,
           0.        , 0.        , 0.        , 0.22222222]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: The local dimension.
   :param alpha: The parameter of the isotropic state.
   :return: Isotropic state of dimension :code:`dim`.




states.chessboard
=================

.. py:module:: states.chessboard

.. autoapi-nested-parse::

   Chessboard state represent the state of a chessboard used in quantum chess.

   In a quantum chessboard, each chess piece is quantum having a superposition of channel states, giving rise to a unique
   chess piece.



Functions
---------

.. autoapisummary::

   states.chessboard.chessboard


Module Contents
---------------

.. py:function:: chessboard(mat_params, s_param = None, t_param = None)

   Produce a chessboard state :cite:`Bruß_2000_Construction`.

   Generates the chessboard state defined in :cite:`Bruß_2000_Construction`. Note that, for certain choices of
   :code:`s_param` and :code:`t_param`, this state will not have positive partial transpose, and
   thus may not be bound entangled.

   .. rubric:: Examples

   The standard chessboard state can be invoked using :code:`|toqito⟩` as

   >>> from toqito.states import chessboard
   >>> chessboard([1, 2, 3, 4, 5, 6], 7, 8)
   array([[ 0.22592593,  0.        ,  0.12962963,  0.        ,  0.        ,
            0.        ,  0.17777778,  0.        ,  0.        ],
          [ 0.        ,  0.01851852,  0.        ,  0.        ,  0.        ,
            0.01111111,  0.        ,  0.02962963,  0.        ],
          [ 0.12962963,  0.        ,  0.18148148,  0.        ,  0.15555556,
            0.        ,  0.        ,  0.        ,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.01851852,  0.        ,
            0.02222222,  0.        , -0.01481481,  0.        ],
          [ 0.        ,  0.        ,  0.15555556,  0.        ,  0.22592593,
            0.        , -0.14814815,  0.        ,  0.        ],
          [ 0.        ,  0.01111111,  0.        ,  0.02222222,  0.        ,
            0.03333333,  0.        ,  0.        ,  0.        ],
          [ 0.17777778,  0.        ,  0.        ,  0.        , -0.14814815,
            0.        ,  0.23703704,  0.        ,  0.        ],
          [ 0.        ,  0.02962963,  0.        , -0.01481481,  0.        ,
            0.        ,  0.        ,  0.05925926,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ]])

   .. rubric:: References

   .. bibliography::
     :filter: docname in docnames

   :param mat_params: Parameters of the chessboard state as defined in :cite:`Bruß_2000_Construction`.
   :param s_param: Default is :code:`np.conj(mat_params[2]) / np.conj(mat_params[5])`.
   :param t_param: Default is :code:`t_param = mat_params[0] * mat_params[3] / mat_params[4]`.
   :return: A chessboard state.




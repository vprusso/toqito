states.tile
===========

.. py:module:: states.tile

.. autoapi-nested-parse::

   Tile state.



Functions
---------

.. autoapisummary::

   states.tile.tile


Module Contents
---------------

.. py:function:: tile(idx)

   Produce a Tile state :cite:`Bennett_1999_UPB`.

   The Tile states constitute five states on 3-by-3 dimensional space that form a UPB (unextendible product basis).

   Returns one of the following five tile states depending on the value of :code:`idx`:

   .. math::
       \begin{equation}
           \begin{aligned}
               |\psi_0 \rangle = \frac{1}{\sqrt{2}} |0 \rangle
               \left(|0\rangle - |1\rangle \right),
               \qquad &
               |\psi_1\rangle = \frac{1}{\sqrt{2}}
               \left(|0\rangle - |1\rangle \right) |2\rangle, \\
               |\psi_2\rangle = \frac{1}{\sqrt{2}} |2\rangle
               \left(|1\rangle - |2\rangle \right),
               \qquad &
               |\psi_3\rangle = \frac{1}{\sqrt{2}}
               \left(|1\rangle - |2\rangle \right) |0\rangle, \\
               \qquad &
               |\psi_4\rangle = \frac{1}{3}
               \left(|0\rangle + |1\rangle + |2\rangle)\right)
               \left(|0\rangle + |1\rangle + |2\rangle \right).
           \end{aligned}
       \end{equation}

   .. rubric:: Examples

   When :code:`idx = 0`, this produces the following tile state

   .. math::
       \frac{1}{\sqrt{2}} |0\rangle \left( |0\rangle - |1\rangle \right).

   Using :code:`|toqito⟩`, we can see that this yields the proper state.

   >>> from toqito.states import tile
   >>> import numpy as np
   >>> tile(0)
   array([[ 0.70710678],
          [-0.70710678],
          [ 0.        ],
          [ 0.        ],
          [ 0.        ],
          [ 0.        ],
          [ 0.        ],
          [ 0.        ],
          [ 0.        ]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: Invalid value for :code:`idx`.
   :param idx: A parameter in [0, 1, 2, 3, 4]
   :return: Tile state.




channel_ops.complementary_channel
=================================

.. py:module:: channel_ops.complementary_channel

.. autoapi-nested-parse::

   Computes the complementary channel/map of a superoperator.



Functions
---------

.. autoapisummary::

   channel_ops.complementary_channel.complementary_channel


Module Contents
---------------

.. py:function:: complementary_channel(kraus_ops)

   Compute the Kraus operators for the complementary map of a quantum channel.

   (Section: Representations and Characterizations of Channels from :cite:`Watrous_2018_TQI`).

   The complementary map is derived from the given quantum channel's Kraus operators by
   rearranging the rows of the input Kraus operators into the Kraus operators of the
   complementary map.

   Specifically, for each Kraus operator :math:`K_i` in the input channel :math:`\Phi`,
   we define the complementary Kraus operators :math:`K_i^C` by stacking the rows of
   :math:`K_i` from all Kraus operators vertically.

   .. rubric:: Examples

   Suppose the following Kraus operators define a quantum channel:

   .. math::
       K_1 = \frac{1}{\sqrt{2}} \begin{pmatrix}
           1 & 0 \\
           0 & 0
       \end{pmatrix},
       K_2 = \frac{1}{\sqrt{2}} \begin{pmatrix}
           0 & 1 \\
           0 & 0
       \end{pmatrix},
       K_3 = \frac{1}{\sqrt{2}} \begin{pmatrix}
           0 & 0 \\
           1 & 0
       \end{pmatrix},
       K_4 = \frac{1}{\sqrt{2}} \begin{pmatrix}
           0 & 0 \\
           0 & 1
       \end{pmatrix}

   To compute the Kraus operators for the complementary map, we rearrange the rows of these
   Kraus operators as follows:

   >>> import numpy as np
   >>> from toqito.channel_ops import complementary_channel
   >>> kraus_ops_Phi = [
   ...     np.sqrt(0.5) * np.array([[1, 0], [0, 0]]),
   ...     np.sqrt(0.5) * np.array([[0, 1], [0, 0]]),
   ...     np.sqrt(0.5) * np.array([[0, 0], [1, 0]]),
   ...     np.sqrt(0.5) * np.array([[0, 0], [0, 1]])
   ... ]
   >>> comp_kraus_ops = complementary_channel(kraus_ops_Phi)
   >>> for i, op in enumerate(comp_kraus_ops):
   ...     print(f"Kraus operator {i + 1}:")
   ...     print(op)
   Kraus operator 1:
   [[0.70710678 0.        ]
    [0.         0.70710678]
    [0.         0.        ]
    [0.         0.        ]]
   Kraus operator 2:
   [[0.         0.        ]
    [0.         0.        ]
    [0.70710678 0.        ]
    [0.         0.70710678]]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If the input is not a valid list of Kraus operators.
   :param kraus_ops: A list of numpy arrays representing the Kraus operators of a quantum channel.
                     Each Kraus operator is assumed to be a square matrix.
   :return: A list of numpy arrays representing the Kraus operators of the complementary map.



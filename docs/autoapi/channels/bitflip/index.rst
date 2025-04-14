channels.bitflip
================

.. py:module:: channels.bitflip

.. autoapi-nested-parse::

   Implements the bitflip quantum gate channel.



Functions
---------

.. autoapisummary::

   channels.bitflip.bitflip


Module Contents
---------------

.. py:function:: bitflip(input_mat = None, prob = 0)

   Apply the bitflip quantum channel to a state or return the Kraus operators.

   The *bitflip channel* is a quantum channel that flips a qubit from :math:`|0\rangle` to :math:`|1\rangle`
   and from :math:`|1\rangle` to :math:`|0\rangle` with probability :math:`p`.
   It is defined by the following operation:

   .. math::

       \mathcal{E}(\rho) = (1-p) \rho + p X \rho X

   where :math:`X` is the Pauli-X (NOT) gate given by:

   .. math::

       X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

   The Kraus operators for this channel are:

   .. math::

       K_0 = \sqrt{1-p} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
       K_1 = \sqrt{p} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

   .. rubric:: Examples

   We can generate the Kraus operators for the bitflip channel with probability 0.3:

   >>> from toqito.channels import bitflip
   >>> bitflip(prob=0.3) # doctest: +NORMALIZE_WHITESPACE
   [array([[0.83666003, 0.        ],
           [0.        , 0.83666003]]),
    array([[0.        , 0.54772256],
           [0.54772256, 0.        ]])]

   We can also apply the bitflip channel to a quantum state. For the state :math:`|0\rangle`,
   the bitflip channel with probability 0.3 produces:

   >>> from toqito.channels import bitflip
   >>> import numpy as np
   >>> rho = np.array([[1, 0], [0, 0]])  # |0><0|
   >>> bitflip(rho, prob=0.3) # doctest: +NORMALIZE_WHITESPACE
   array([[0.7+0.j, 0. +0.j],
       [0. +0.j, 0.3+0.j]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param input_mat: A matrix or state to apply the channel to. If `None`, returns the Kraus operators.
   :param prob: The probability of a bitflip occurring.
   :return: Either the Kraus operators of the bitflip channel if `input_mat` is `None`,
            or the result of applying the channel to `input_mat`.



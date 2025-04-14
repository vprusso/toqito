channel_props.is_extremal
=========================

.. py:module:: channel_props.is_extremal

.. autoapi-nested-parse::

   Determines whether a quantum channel is extremal.



Functions
---------

.. autoapisummary::

   channel_props.is_extremal.is_extremal


Module Contents
---------------

.. py:function:: is_extremal(phi, tol = 1e-09)

   Determine whether a quantum channel is extremal.

   (Section 2.2.4: Extremal Channels from :cite:`Watrous_2018_TQI`).

   Theorem 2.31 in :cite:`Watrous_2018_TQI` provides the characterization of extremal
   quantum channels as a channel :math:`\Phi` is an extreme point of the convex set
   of quantum channels if and only if the collection:

   .. math::
       \{ A_i^\dagger A_j \}_{i,j=1}^{r}

   is linearly independent.

   The channel can be provided in one of the following representations:

   - A Choi matrix, representing the quantum channel in the Choi representation. It will
     be converted internally to a set of Kraus operators.
   - A list of Kraus operators, representing the channel in Kraus form.
   - A nested list of Kraus operators, which will be flattened automatically.

   .. rubric:: Examples

   The following demonstrates an example of an extremal quantum channel from Example 2.33
   in :cite:`Watrous_2018_TQI`.

   >>> import numpy as np
   >>> from toqito.channel_props import is_extremal
   >>> kraus_ops = [
   ...     (1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]]),
   ...     (1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]])
   ... ]
   >>> is_extremal(kraus_ops)
   np.True_

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: The quantum channel, which may be given as a Choi matrix or a list of Kraus operators.
   :param tol: Tolerance value for numerical precision in rank computation.
   :type phi: list[numpy.ndarray] | list[list[numpy.ndarray]] | numpy.ndarray
   :raises ValueError: If the input is neither a valid list of Kraus operators nor a Choi matrix.
   :return: True if the channel is extremal; False otherwise.



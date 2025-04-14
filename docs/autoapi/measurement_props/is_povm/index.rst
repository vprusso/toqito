measurement_props.is_povm
=========================

.. py:module:: measurement_props.is_povm

.. autoapi-nested-parse::

   Determine if a list of matrices are POVM elements.



Functions
---------

.. autoapisummary::

   measurement_props.is_povm.is_povm


Module Contents
---------------

.. py:function:: is_povm(mat_list)

   Determine if a list of matrices constitute a valid set of POVMs :cite:`WikiPOVM`.

   A valid set of measurements are defined by a set of positive semidefinite operators

   .. math::
        \{P_a : a \in \Gamma\} \subset \text{Pos}(\mathcal{X}),

   indexed by the alphabet :math:`\Gamma` of measurement outcomes satisfying the constraint that

   .. math::
       \sum_{a \in \Gamma} P_a = I_{\mathcal{X}}.

   .. rubric:: Examples

   Consider the following matrices:

   .. math::
       M_0 =
       \begin{pmatrix}
           1 & 0 \\
           0 & 0
       \end{pmatrix}
       \quad \text{and} \quad
       M_1 =
       \begin{pmatrix}
           0 & 0 \\
           0 & 1
       \end{pmatrix}.

   Our function indicates that this set of operators constitute a set of
   POVMs.

   >>> from toqito.measurement_props import is_povm
   >>> import numpy as np
   >>> meas_1 = np.array([[1, 0], [0, 0]])
   >>> meas_2 = np.array([[0, 0], [0, 1]])
   >>> meas = [meas_1, meas_2]
   >>> is_povm(meas)
   True

   We may also use the :code:`random_povm` function from :code:`|toqito⟩`, and can verify that a
   randomly generated set satisfies the criteria for being a POVM set.

   >>> from toqito.measurement_props import is_povm
   >>> from toqito.rand import random_povm
   >>> import numpy as np
   >>> dim, num_inputs, num_outputs = 2, 2, 2
   >>> measurements = random_povm(dim, num_inputs, num_outputs)
   >>> is_povm([measurements[:, :, 0, 0], measurements[:, :, 0, 1]])
   True

   Alternatively, the following matrices

   .. math::
       M_0 =
       \begin{pmatrix}
           1 & 2 \\
           3 & 4
       \end{pmatrix}
       \quad \text{and} \quad
       M_1 =
       \begin{pmatrix}
           5 & 6 \\
           7 & 8
       \end{pmatrix},

   do not constitute a POVM set.

   >>> from toqito.measurement_props import is_povm
   >>> import numpy as np
   >>> non_meas_1 = np.array([[1, 2], [3, 4]])
   >>> non_meas_2 = np.array([[5, 6], [7, 8]])
   >>> non_meas = [non_meas_1, non_meas_2]
   >>> is_povm(non_meas)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat_list: A list of matrices.
   :return: Return :code:`True` if set of matrices constitutes a set of
            measurements, and :code:`False` otherwise.




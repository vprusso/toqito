state_props.is_separable
========================

.. py:module:: state_props.is_separable

.. autoapi-nested-parse::

   Checks if a quantum state is a separable state.



Functions
---------

.. autoapisummary::

   state_props.is_separable.is_separable


Module Contents
---------------

.. py:function:: is_separable(state, dim = None, level = 2, tol = 1e-08)

   Determine if a given state (given as a density matrix) is a separable state :cite:`WikiSepSt`.

   .. rubric:: Examples

   Consider the following separable (by construction) state:

   .. math::
       \rho = \rho_1 \otimes \rho_2.
       \rho_1 = \frac{1}{2} \left(
           |0 \rangle \langle 0| + |0 \rangle \langle 1| + |1 \rangle \langle 0| + |1 \rangle \langle 1| \right)
       \rho_2 = \frac{1}{2} \left( |0 \rangle \langle 0| + |1 \rangle \langle 1| \right)

   The resulting density matrix will be:

   .. math::
       \rho =  \frac{1}{4} \begin{pmatrix}
               1 & 0 & 1 & 0 \\
               0 & 1 & 0 & 1 \\
               1 & 0 & 1 & 0 \\
               0 & 1 & 0 & 1
               \end{pmatrix} \in \text{D}(\mathcal{X}).

   We provide the input as a density matrix :math:`\rho`.

   On the other hand, a random density matrix will be an entangled state (a separable state).

   >>> import numpy as np
   >>> from toqito.rand.random_density_matrix import random_density_matrix
   >>> from toqito.state_props.is_separable import is_separable
   >>> rho_separable = np.array([[1, 0, 1, 0],
   ...                           [0, 1, 0, 1],
   ...                           [1, 0, 1, 0],
   ...                           [0, 1, 0, 1]])
   >>> is_separable(rho_separable)
   True

   >>> rho_not_separable = np.array([[ 0.13407875+0.j        , -0.08263926-0.17760437j,
   ...    -0.0135111 -0.12352182j,  0.0368423 -0.05563985j],
   ...   [-0.08263926+0.17760437j,  0.53338542+0.j        ,
   ...     0.19782968-0.04549732j,  0.11287093+0.17024249j],
   ...   [-0.0135111 +0.12352182j,  0.19782968+0.04549732j,
   ...     0.21254612+0.j        , -0.00875865+0.11144344j],
   ...   [ 0.0368423 +0.05563985j,  0.11287093-0.17024249j,
   ...    -0.00875865-0.11144344j,  0.11998971+0.j        ]])
   >>> is_separable(rho_not_separable)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If dimension is not specified.
   :param state: The matrix to check.
   :param dim: The dimension of the input.
   :param level: The level up to which to search for the symmetric extensions.
   :param tol: Numerical tolerance used.
   :return: :code:`True` if :code:`rho` is separabale and :code:`False` otherwise.




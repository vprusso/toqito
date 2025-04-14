matrices.gen_gell_mann
======================

.. py:module:: matrices.gen_gell_mann

.. autoapi-nested-parse::

   Produces the generalized Gell-Mann operator matrices.



Functions
---------

.. autoapisummary::

   matrices.gen_gell_mann.gen_gell_mann


Module Contents
---------------

.. py:function:: gen_gell_mann(ind_1, ind_2, dim)

   Produce a generalized Gell-Mann operator :cite:`WikiGellMann`.

   Construct a :code:`dim`-by-:code:`dim` Hermitian operator. These matrices
   span the entire space of :code:`dim`-by-:code:`dim` matrices as
   :code:`ind_1` and :code:`ind_2` range from 0 to :code:`dim-1`, inclusive,
   and they generalize the Pauli operators when :code:`dim = 2` and the
   Gell-Mann operators when :code:`dim = 3`.

   .. rubric:: Examples

   The generalized Gell-Mann matrix for :code:`ind_1 = 0`, :code:`ind_2 = 1`
   and :code:`dim = 2` is given as

   .. math::
       G_{0, 1, 2} = \begin{pmatrix}
                        0 & 1 \\
                        1 & 0
                     \end{pmatrix}.

   This can be obtained in :code:`|toqito⟩` as follows.

   >>> from toqito.matrices import gen_gell_mann
   >>> gen_gell_mann(0, 1, 2)
   array([[0., 1.],
          [1., 0.]])

   The generalized Gell-Mann matrix :code:`ind_1 = 2`, :code:`ind_2 = 3`, and
   :code:`dim = 4` is given as

   .. math::
       G_{2, 3, 4} = \begin{pmatrix}
                       0 & 0 & 0 & 0 \\
                       0 & 0 & 0 & 0 \\
                       0 & 0 & 0 & 1 \\
                       0 & 0 & 1 & 0
                     \end{pmatrix}.

   This can be obtained in :code:`|toqito⟩` as follows.

   >>> from toqito.matrices import gen_gell_mann
   >>> gen_gell_mann(2, 3, 4)
   array([[0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 1.],
          [0., 0., 1., 0.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param ind_1: A non-negative integer from 0 to :code:`dim-1` (inclusive).
   :param ind_2: A non-negative integer from 0 to :code:`dim-1` (inclusive).
   :param dim: The dimension of the Gell-Mann operator.
   :return: The generalized Gell-Mann operator as an array.




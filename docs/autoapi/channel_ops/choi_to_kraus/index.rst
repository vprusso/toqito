channel_ops.choi_to_kraus
=========================

.. py:module:: channel_ops.choi_to_kraus

.. autoapi-nested-parse::

   Computes a list of Kraus operators from the Choi matrix.



Functions
---------

.. autoapisummary::

   channel_ops.choi_to_kraus.choi_to_kraus


Module Contents
---------------

.. py:function:: choi_to_kraus(choi_mat, tol = 1e-09, dim = None)

   Compute a list of Kraus operators from the Choi matrix from :cite:`Rigetti_2022_Forest`.

   Note that unlike the Choi or natural representation of operators, the Kraus representation is
   *not* unique.

   If the input channel maps :math:`M_{r,c}` to :math:`M_{x,y}` then :code:`dim` should be the
   list :code:`[[r,x], [c,y]]`. If it maps :math:`M_m` to :math:`M_n`, then :code:`dim` can simply
   be the vector :code:`[m,n]`.

   For completely positive maps the output is a single flat list of numpy arrays since the left and
   right Kraus maps are the same.

   This function has been adapted from :cite:`Rigetti_2022_Forest` and QETLAB :cite:`QETLAB_link`.

   .. rubric:: Examples

   Consider taking the Kraus operators of the Choi matrix that characterizes the "swap operator"
   defined as

   .. math::
       \begin{pmatrix}
           1 & 0 & 0 & 0 \\
           0 & 0 & 1 & 0 \\
           0 & 1 & 0 & 0 \\
           0 & 0 & 0 & 1
       \end{pmatrix}

   The corresponding Kraus operators of the swap operator are given as follows,

   .. math::
       \begin{equation}
       \big[
           \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},
           \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
       \big],
       \big[
           \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},
           \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
       \big],
       \big[
           \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix},
           \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
       \big],
       \big[
           \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix},
           \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}
       \big]
       \end{equation}

   This can be verified in :code:`|toqito⟩` as follows.

   >>> import numpy as np
   >>> from toqito.channel_ops import choi_to_kraus
   >>> choi_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
   >>> kraus_ops = choi_to_kraus(choi_mat)
   >>> kraus_ops
   [[array([[ 0.        ,  0.70710678],
          [-0.70710678,  0.        ]]), array([[-0.        , -0.70710678],
          [ 0.70710678, -0.        ]])], [array([[0.        , 0.70710678],
          [0.70710678, 0.        ]]), array([[0.        , 0.70710678],
          [0.70710678, 0.        ]])], [array([[1., 0.],
          [0., 0.]]), array([[1., 0.],
          [0., 0.]])], [array([[0., 0.],
          [0., 1.]]), array([[0., 0.],
          [0., 1.]])]]

   .. seealso:: :func:`.kraus_to_choi`

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param choi_mat: A Choi matrix
   :param tol: optional threshold parameter for eigenvalues/kraus ops to be discarded
   :param dim: A scalar, vector or matrix containing the input and output dimensions of Choi matrix.
   :return: List of Kraus operators




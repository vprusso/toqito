matrix_props.is_orthonormal
===========================

.. py:module:: matrix_props.is_orthonormal

.. autoapi-nested-parse::

   Checks if the set of vectors are orthonormal.



Functions
---------

.. autoapisummary::

   matrix_props.is_orthonormal.is_orthonormal


Module Contents
---------------

.. py:function:: is_orthonormal(vectors)

   Check if the vectors are orthonormal.

   .. rubric:: Examples

   The following vectors are an example of an orthonormal set of
   vectors in :math:`\mathbb{R}^3`.

   .. math::
       \begin{pmatrix}
           1 \\ 0 \\ 1
       \end{pmatrix}, \quad
       \begin{pmatrix}
           1 \\ 1 \\ 0
       \end{pmatrix}, \quad \text{and} \quad
       \begin{pmatrix}
           0 \\ 0 \\ 1
       \end{pmatrix}

   To check these are a known set of orthonormal vectors:

   >>> import numpy as np
   >>> from toqito.matrix_props import is_orthonormal
   >>> v_1 = np.array([1, 0, 0])
   >>> v_2 = np.array([0, 1, 0])
   >>> v_3 = np.array([0, 0, 1])
   >>> v = np.array([v_1, v_2, v_3])
   >>> is_orthonormal(v)
   True

   :param vectors: A list of `np.ndarray` 1-by-n vectors.
   :return: True if vectors are orthonormal; False otherwise.



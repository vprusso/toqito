state_props.is_npt
==================

.. py:module:: state_props.is_npt

.. autoapi-nested-parse::

   Checks if the quantum state has NPT (negative partial transpose) criterion.



Functions
---------

.. autoapisummary::

   state_props.is_npt.is_npt


Module Contents
---------------

.. py:function:: is_npt(mat, sys = 2, dim = None, tol = None)

   Determine whether or not a matrix has negative partial transpose :cite:`WikiPeresHorodecki`.

   Yields either :code:`True` or :code:`False`, indicating that :code:`mat` does or does not have
   negative partial transpose (within numerical error). The variable :code:`mat` is assumed to act
   on bipartite space. :cite:`DiVincenzo_2000_Evidence`

   A state has negative partial transpose if it does not have positive partial transpose.

   .. rubric:: Examples

   To check if a matrix has negative partial transpose

   >>> import numpy as np
   >>> from toqito.state_props import is_npt
   >>> from toqito.states import bell
   >>> is_npt(bell(2) @ bell(2).conj().T, 2)
   True

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: A square matrix.
   :param sys: Scalar or vector indicating which subsystems the transpose
               should be applied on. Default value is `2`.
   :param dim: The dimension is a vector containing the dimensions of the
               subsystems on which :code:`mat` acts.
   :param tol: Tolerance with which to check whether `mat` is PPT.
   :return: Returns :code:`True` if :code:`mat` is NPT and :code:`False` if
            not.




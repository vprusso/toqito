states.bb84
===========

.. py:module:: states.bb84

.. autoapi-nested-parse::

   BB84 states represent the BB84 basis states, which are based on BB84, a quantum key distribution scheme.

   In the BB884 scheme, each qubit is encoded with one of the 4 polarization states: 0, 1, +45° or -45°.



Functions
---------

.. autoapisummary::

   states.bb84.bb84


Module Contents
---------------

.. py:function:: bb84()

   Obtain the BB84 basis states :cite:`WikiBB84`.

   The BB84 basis states are defined as

   .. math::
       |0\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \\
       |1\rangle := \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad \\
       |+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \\
       |-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}.

   .. rubric:: Examples

   The BB84 basis states can be obtained in :code:`|toqito⟩` as follows in the form of a list of
   arrays.

   >>> from toqito.states import bb84
   >>> bb84()
   [[array([[1.],
          [0.]]), array([[0.],
          [1.]])], [array([[0.70710678],
          [0.70710678]]), array([[ 0.70710678],
          [-0.70710678]])]]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :return: The four BB84 basis states.




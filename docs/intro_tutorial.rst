Introductory Tutorial
======================

This tutorial will illustrate the basics of how to use :code:`toqito`. This
will cover how to instantiate and use the fundamental objects that
:code:`toqito` provides; namely quantum states, channels, and measurements.

This is a user guide for :code:`toqito` and is not meant to serve as an
introduction to quantum information. For introductory material on quantum
information, please consult "Quantum Information and Quantum Computation" by
Nielsen and Chuang or the freely available lecture notes `"Introduction to
Quantum Computing" <https://cs.uwaterloo.ca/~watrous/QC-notes/>`_
by John Watrous.

More advanced tutorials can be found on the `tutorials page <https://toqito.readthedocs.io/en/latest/tutorials.html>`_.

This tutorial assumes you have :code:`toqito` installed on your machine. If you
do not, please consult the installation instructions in :ref:`getting_started_reference-label`.

States
------

A *quantum state* is a density operator

.. math::
    \rho \in \text{D}(\mathcal{X})

where :math:`\mathcal{X}` is a complex Euclidean space and where
:math:`\text{D}(\cdot)` represents the set of density matrices, that is, the
set of matrices that are positive semidefinite with trace equal to :math:`1`.

Quantum States
^^^^^^^^^^^^^^

A complete overview of the scope of quantum states can be found
`here <https://toqito.readthedocs.io/en/latest/autoapi/states/index.html>`_

The standard basis ket vectors given as :math:`|0\rangle` and :math:`|1\rangle` where

.. math::
    | 0 \rangle = [1, 0]^{\text{T}} \quad \text{and} \quad | 1 \rangle = [0, 1]^{\text{T}}

can be defined in :code:`toqito` as such

.. code-block:: python

    >>> from toqito.states import basis
    >>> # |0>
    >>> basis(2, 0)
    array([[1],
           [0]])

    >>> # |1>
    >>> basis(2, 1)
    array([[0],
           [1]])


One may define one of the four Bell states written as

.. math::
    u_0 = \frac{1}{\sqrt{2}} \left(| 00 \rangle + | 11 \rangle \right)

using :code:`toqito` as

.. code-block:: python

    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> u_0 = 1/np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    >>> u_0
    array([[0.70710678],
           [0.        ],
           [0.        ],
           [0.70710678]])


The corresponding density operator of :math:`u_0` can be obtained from

.. math::
    \rho_0 = u_0 u_0^* = \frac{1}{2} 
    \begin{pmatrix} 
        1 & 0 & 0 & 1 \\
        0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 \\
        1 & 0 & 0 & 1
    \end{pmatrix} \in \text{D}(\mathcal{X}).

In :code:`toqito`, that can be obtained as 

.. code-block:: python
    
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> u_0 = 1/np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    >>> rho_0 = u_0 @ u_0.conj().T
    >>> rho_0
    array([[0.5, 0. , 0. , 0.5],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0.5, 0. , 0. , 0.5]])


Alternatively, we may leverage the :code:`bell` function in :code:`toqito` to
generate all four Bell states defined as

.. math::
    \begin{equation}
        \begin{aligned}
            u_0 = \frac{1}{\sqrt{2}} \left(| 00 \rangle + | 11 \rangle \right), &\quad 
            u_1 = \frac{1}{\sqrt{2}} \left(| 00 \rangle - | 11 \rangle \right), \\
            u_2 = \frac{1}{\sqrt{2}} \left(| 01 \rangle + | 10 \rangle \right), &\quad
            u_3 = \frac{1}{\sqrt{2}} \left(| 01 \rangle - | 10 \rangle \right),
        \end{aligned}
    \end{equation}

in a more concise manner as 

.. code-block:: python

    >>> from toqito.states import bell
    >>> import numpy as np
    >>> bell(0)
    array([[0.70710678],
           [0.        ],
           [0.        ],
           [0.70710678]])

The Bell states constitute one such well-known class of quantum states. There
are many other classes of states that are widely used in the field of quantum
information. For instance, the GHZ state 

.. math::
    | GHZ \rangle = \frac{1}{\sqrt{2}} \left( | 000 \rangle + | 111 \rangle \right)

is a well-known 3-qubit quantum state. We can invoke this using :code:`toqito` as

.. code-block:: python

    >>> from toqito.states import ghz
    >>> ghz(2, 3)
    array([[0.70710678],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.70710678]])


While the 3-qubit form of the GHZ state is arguably the most notable, it is
possible to define a generalized GHZ state

.. math::
    | GHZ_n \rangle = \frac{1}{\sqrt{n}} \left( | 0 \rangle^{\otimes n} + | 1
    \rangle^{\otimes n} \right).

This generalized state may be obtained in :code:`toqito` as well. For instance,
here is the GHZ state :math:`\mathbb{C}^{4^{\otimes 7}}` as 

.. math::
    \frac{1}{\sqrt{30}} \left(| 0000000 \rangle + 2| 1111111 \rangle + 3|
    2222222 \rangle + 4| 3333333\rangle \right).


Properties of Quantum States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a quantum state, it is often useful to be able to determine certain
*properties* of the state.

For instance, we can check if a quantum state is pure, that is, if the density
matrix that describes the state has rank 1.

Any one of the Bell states serve as an example of a pure state

.. code-block:: python

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> rho = bell(0) @ bell(0).conj().T
    >>> is_pure(rho)
    True

Another property that is useful is whether a given state is PPT (positive
partial transpose), that is, whether the state remains positive after taking
the partial transpose of the state.

For quantum states consisting of shared systems of either dimension :math:`2
\otimes 2` or :math:`2 \otimes 3`, the notion of whether a state is PPT serves
as a method to determine whether a given quantum state is entangled or
separable.

As an example, any one of the Bell states constitute a canonical maximally
entangled state over :math:`2 \otimes 2` and therefore should not satisfy the
PPT criterion.

.. code-block:: python

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_ppt
    >>> rho = bell(2) @ bell(2).conj().T
    >>> is_ppt(rho)
    False

As we can see, the PPT criterion is :code:`False` for an entangled state in
:math:`2 \otimes 2`.

Determining whether a quantum state is separable or entangled is often useful
but is, unfortunately, NP-hard. For a given density matrix represented by a
quantum state, we can use :code:`toqito` to run a number of separability tests
from the literature to determine if it is separable or entangled. 

For instance, the following bound-entangled tile state is found to be entangled
(i.e. not separable).

.. code-block:: python

    >>> import numpy as np
    >>> from toqito.state_props import is_separable
    >>> from toqito.states import tile
    >>> rho = np.identity(9)
    >>> for i in range(5):
    ...    rho = rho - tile(i) @ tile(i).conj().T
    >>> rho = rho / 4
    >>> is_separable(rho)
    False

Further properties that one can check via :code:`toqito` may be found `on this page
<https://toqito.readthedocs.io/en/latest/autoapi/state_props/index.html>`_.

Distance Metrics for Quantum States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given two quantum states, it is often useful to have some way in which to
quantify how similar or different one state is from another.

One well known metric is the *fidelity* function defined for two quantum
states. For two states :math:`\rho` and :math:`\sigma`, one defines the
fidelity between :math:`\rho` and :math:`\sigma` as 

.. math::
    || \sqrt{\rho} \sqrt{\sigma} ||_1,

where :math:`|| \cdot ||_1` denotes the trace norm. 

The fidelity function yields a value between :math:`0` and :math:`1`, with
:math:`0` representing the scenario where :math:`\rho` and :math:`\sigma` are
as different as can be and where a value of :math:`1` indicates a scenario
where :math:`\rho` and :math:`\sigma` are identical.

Let us consider an example in :code:`toqito` where we wish to calculate the
fidelity function between quantum states that happen to be identical.

.. code-block:: python

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import fidelity
    >>> import numpy as np
    >>>
    >>> # Define two identical density operators.
    >>> rho = bell(0)*bell(0).conj().T
    >>> sigma = bell(0)*bell(0).conj().T
    >>> 
    >>> # Calculate the fidelity between `rho` and `sigma`
    >>> np.around(fidelity(rho, sigma), decimals=2)
    np.float64(1.0)

There are a number of other metrics one can compute on two density matrices
including the trace norm, trace distance. These and others are also available
in :code:`toqito`. For a full list of distance metrics one can compute on
quantum states, consult the docs.

Channels
--------

A *quantum channel* can be defined as a completely positive and trace
preserving linear map.

More formally, let :math:`\mathcal{X}` and :math:`\mathcal{Y}` represent
complex Euclidean spaces and let :math:`\text{L}(\cdot)` represent the set of
linear operators. Then a quantum channel, :math:`\Phi` is defined as

.. math::
    \Phi: \text{L}(\mathcal{X}) \rightarrow \text{L}(\mathcal{Y})

such that :math:`\Phi` is completely positive and trace preserving.

Quantum Channels
^^^^^^^^^^^^^^^^

The partial trace operation is an often used in various applications of quantum
information. The partial trace is defined as

    .. math::
        \left( \text{Tr} \otimes \mathbb{I}_{\mathcal{Y}} \right)
        \left(X \otimes Y \right) = \text{Tr}(X)Y

where :math:`X \in \text{L}(\mathcal{X})` and :math:`Y \in
\text{L}(\mathcal{Y})` are linear operators over complex Euclidean spaces
:math:`\mathcal{X}` and :math:`\mathcal{Y}`.

Consider the following matrix

.. math::
    X = \begin{pmatrix}
            1 & 2 & 3 & 4 \\
            5 & 6 & 7 & 8 \\
            9 & 10 & 11 & 12 \\
            13 & 14 & 15 & 16
        \end{pmatrix}.

Taking the partial trace over the second subsystem of :math:`X` yields the following matrix

.. math::
    X_{pt, 2} = \begin{pmatrix}
                7 & 11 \\
                23 & 27
                \end{pmatrix}.

By default, the partial trace function in :code:`toqito` takes the trace of the second
subsystem.

.. code-block:: python

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.array(
    ...     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ... )
    >>> partial_trace(test_input_mat)
    array([[ 7, 11],
           [23, 27]])


By specifying the :code:`sys = [0]` argument, we can perform the partial trace over the first
subsystem (instead of the default second subsystem as done above). Performing the partial
trace over the first subsystem yields the following matrix

.. math::
    X_{pt, 1} = \begin{pmatrix}
                    12 & 14 \\
                    20 & 22
                \end{pmatrix}.

.. code-block:: python

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.array(
    ...     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ... )
    >>> partial_trace(test_input_mat, [0])
    array([[12, 14],
           [20, 22]])



Another often useful channel is the *partial transpose*. The *partial transpose*
is defined as

    .. math::
        \left( \text{T} \otimes \mathbb{I}_{\mathcal{Y}} \right)
        \left(X\right)

where :math:`X \in \text{L}(\mathcal{X})` is a linear operator over the complex
Euclidean space :math:`\mathcal{X}` and where :math:`\text{T}` is the transpose
mapping :math:`\text{T} \in \text{T}(\mathcal{X})` defined as

.. math::
    \text{T}(X) = X^{\text{T}}

for all :math:`X \in \text{L}(\mathcal{X})`.

Consider the following matrix

.. math::
    X = \begin{pmatrix}
            1 & 2 & 3 & 4 \\
            5 & 6 & 7 & 8 \\
            9 & 10 & 11 & 12 \\
            13 & 14 & 15 & 16
        \end{pmatrix}.

Performing the partial transpose on the matrix :math:`X` over the second
subsystem yields the following matrix

.. math::
    X_{pt, 2} = \begin{pmatrix}
                1 & 5 & 3 & 7 \\
                2 & 6 & 4 & 8 \\
                9 & 13 & 11 & 15 \\
                10 & 14 & 12 & 16
                \end{pmatrix}.

By default, in :code:`toqito`, the partial transpose function performs the transposition on
the second subsystem as follows.

.. code-block:: python

    >>> from toqito.channels import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.arange(1, 17).reshape(4, 4)
    >>> partial_transpose(test_input_mat)
    array([[ 1,  5,  3,  7],
           [ 2,  6,  4,  8],
           [ 9, 13, 11, 15],
           [10, 14, 12, 16]])


By specifying the :code:`sys = [0]` argument, we can perform the partial transpose over the
first subsystem (instead of the default second subsystem as done above). Performing the
partial transpose over the first subsystem yields the following matrix

.. math::
    X_{pt, 1} = \begin{pmatrix}
                    1 & 2 & 9 & 10 \\
                    5 & 6 & 13 & 14 \\
                    3 & 4 & 11 & 12 \\
                    7 & 8 & 15 & 16
                \end{pmatrix}.
  
.. code-block:: python

    >>> from toqito.channels import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.array(
    ...     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ... )
    >>> partial_transpose(test_input_mat, [0])
    array([[ 1,  2,  9, 10],
           [ 5,  6, 13, 14],
           [ 3,  4, 11, 12],
           [ 7,  8, 15, 16]])



Measurements
------------

A *measurement* can be defined as a function

.. math::
    \mu: \Sigma \rightarrow \text{Pos}(\mathcal{X})

satisfying

.. math::
    \sum_{a \in \Sigma} \mu(a) = \mathbb{I}_{\mathcal{X}}

where :math:`\Sigma` represents a set of measurement outcomes and where
:math:`\mu(a)` represents the measurement operator associated with outcome
:math:`a \in \Sigma`.

POVM
^^^^

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

A POVM (Positive Operator-Valued Measure) is a set of positive operators that sum to the identity. 
Our function expects this set of operators to be a POVM because it checks if the sum of the operators 
equals the identity, ensuring that the measurement outcomes are properly normalized.

    >>> from toqito.measurement_props import is_povm
    >>> import numpy as np
    >>> meas_1 = np.array([[1, 0], [0, 0]])
    >>> meas_2 = np.array([[0, 0], [0, 1]])
    >>> meas = [meas_1, meas_2]
    >>> is_povm(meas)
    True

Random POVM
^^^^^^^^^^^

A POVM (Positive Operator-Valued Measure) is a set of positive semidefinite operators {M₁, M₂, ..., Mₙ} 
that sum to the identity matrix (∑ᵢMᵢ = I). Each operator Mᵢ corresponds to a possible measurement outcome, 
and the probability of obtaining outcome i when measuring a quantum state ρ is given by Tr(Mᵢρ). 
POVMs provide the most general description of quantum measurements allowed by quantum mechanics.

We may also use the :code:`random_povm` function from :code:`toqito`, and can verify that a
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

Measure
^^^^^^^

Consider the following state:

.. math::
    u = \frac{1}{\sqrt{3}} e_0 + \sqrt{\frac{2}{3}} e_1

where we define :math:`u u^* = \rho \in \text{D}(\mathcal{X})`.

Let :math:`e_0` and :math:`e_1` be the standard basis vectors:

.. math::
    e_0 = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \quad \text{and} \quad e_1 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}

Define measurement operators

.. math::
    P_0 = e_0 e_0^* \quad \text{and} \quad P_1 = e_1 e_1^*.

.. code-block:: python

    >>> from toqito.states import basis
    >>> from toqito.measurement_ops import measure
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>>
    >>> u = 1/np.sqrt(3) * e_0 + np.sqrt(2/3) * e_1
    >>> rho = u @ u.conj().T
    >>>
    >>> proj_0 = e_0 @ e_0.conj().T
    >>> proj_1 = e_1 @ e_1.conj().T

Then the probability of obtaining outcome :math:`0` is given by

.. math::
    \langle P_0, \rho \rangle = \frac{1}{3}.

.. code-block:: python

    >>> measure(proj_0, rho)
    0.3333333333333334

Similarly, the probability of obtaining outcome :math:`1` is given by

.. math::
    \langle P_1, \rho \rangle = \frac{2}{3}.

.. code-block:: python

    >>> measure(proj_1, rho)
    0.6666666666666666

Pretty Good Measurement
^^^^^^^^^^^^^^^^^^^^^^^

Consider "pretty good measurement" on the set of trine states.

The pretty good measurement (PGM), also known as the "square root measurement" is a set of POVMs :math:`(G_1, \ldots, G_n)` defined as

.. math::
    G_i = P^{-1/2} \left(p_i \rho_i\right) P^{-1/2} \quad \text{where} \quad P = \sum_{i=1}^n p_i \rho_i.

This measurement was initially defined in :cite:`Hughston_1993_Complete` and has found applications in quantum state discrimination tasks. 
While not always optimal, the PGM provides a reasonable measurement strategy that can be computed efficiently.

For example, consider the following trine states:

.. math::
    u_0 = |0\rangle, \quad
    u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
    u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

.. code-block:: python

    >>> from toqito.states import trine
    >>> from toqito.measurements import pretty_good_measurement
    >>>
    >>> states = trine()
    >>> probs = [1 / 3, 1 / 3, 1 / 3]
    >>> pgm = pretty_good_measurement(states, probs)
    >>> pgm
    [array([[0.66666667, 0.        ],
           [0.        , 0.        ]]), array([[0.16666667, 0.28867513],
           [0.28867513, 0.5       ]]), array([[ 0.16666667, -0.28867513],
           [-0.28867513,  0.5       ]])]

Pretty Bad Measurement
^^^^^^^^^^^^^^^^^^^^^^

Similarly, we can consider so-called "pretty bad measurement" (PBM) on the set of trine states :cite:`McIrvin_2024_Pretty`.

.. math::
    u_0 = |0\rangle, \quad
    u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
    u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

.. code-block:: python

    >>> from toqito.states import trine
    >>> from toqito.measurements import pretty_bad_measurement
    >>>
    >>> states = trine()
    >>> probs = [1 / 3, 1 / 3, 1 / 3]
    >>> pbm = pretty_bad_measurement(states, probs)
    >>> pbm
    [array([[0.16666667, 0.        ],
           [0.        , 0.5       ]]), array([[ 0.41666667, -0.14433757],
           [-0.14433757,  0.25      ]]), array([[0.41666667, 0.14433757],
           [0.14433757, 0.25      ]])]

References
------------------------------

.. bibliography:: 
    :filter: docname in docnames
Introductory Tutorial
======================

This tutorial will illustrate the basics of how to use :code:`toqito`. This
will cover how to instantiate and use the fundamental objects that
:code:`toqito` provides; namely quantum states, channels, and measurements.

This is a user guide for :code:`toqito` and is not meant to serve as an
introduction to quantum information. For introductory material on quantum
information, please consult "Quantum Information and Quantum Computation" by
Nielsen and Chuang or the freely available lecture notes `"Introduction to
Quantum Computing"
<https://cs.uwaterloo.ca/~watrous/LectureNotes/CPSC519.Winter2006/all.pdf)>`_
by John Watrous.

More advanced tutorials can be found on the `tutorials page
<https://toqito.readthedocs.io/en/latest/tutorials.html>`_.

This tutorial assumes you have :code:`toqito` installed on your machine. If you
do not, please consult the `installation instructions
<https://toqito.readthedocs.io/en/latest/install.html>`_.

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
`here <https://toqito.readthedocs.io/en/latest/states.html#quantum-states>`_

The standard basis bra vectors given as :math:`|0\rangle` and :math:`|1\rangle` where

.. math::
    | 0 \rangle = [1, 0]^{\text{T}} \quad \text{and} \quad | 1 \rangle = [0, 1]^{\text{T}}

can be defined in :code:`toqito` as such

.. code-block:: python

    >>> from toqito.states import basis
    >>> # |0>
    >>> basis(2, 0)
    [[1]
    [0]]
    >>> # |1>
    >>> basis(2, 1)
    [[0]
    [1]]

One may define one of the four Bell states written as

.. math::
    u_0 = \frac{1}{\sqrt{2}} \left(| 00 \rangle + | 11 \rangle \right)

using :code:`toqito` as

.. code-block:: python

    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> u_0 = 1/np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.70710678]]

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
    
    >>> rho_0 = u_0 * u_0.conj().T
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]
     [0.5 0.  0.  0.5]]

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
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.70710678]]

The Bell states constitute one such well-known class of quantum states. There
are many other classes of states that are widely used in the field of quantum
information. For instance, the GHZ state 

.. math::
    | GHZ \rangle = \frac{1}{\sqrt{2}} \left( | 000 \rangle + | 111 \rangle \right)

is a well-known 3-qubit quantum state. We can invoke this using :code:`toqito` as

.. code-block:: python

    >>> from toqito.states import ghz
    >>> ghz(2, 3).toarray()
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.70710678]]

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

Using :code:`toqito`, we can see this generates the appropriate generalized GHZ
state.

.. code-block:: python

    >>> from toqito.states import ghz
    >>> ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30)).toarray()
    [[0.18257419],
     [0.        ],
     [0.        ],
     ...,
     [0.        ],
     [0.        ],
     [0.73029674]])

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
    >>> rho = bell(0) * bell(0).conj().T
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
    >>> rho = bell(2) * bell(2).conj().T
    >>> is_ppt(rho)
    False

As we can see, the PPT criterion is :code:`False` for an entangled state in
:math:`2 \otimes 2`.

Further properties that one can check via :code:`toqito` may be found `on this page
<https://toqito.readthedocs.io/en/latest/states.html#properties-of-quantum-states>`_.


Operations on Quantum States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

The fidelity fucntion yields a value between :math:`0` and :math:`1`, with
:math:`0` representing the scenarion where :math:`\rho` and :math:`\sigma` are
as different as can be and where a value of :math:`1` indicates a scenario
where :math:`\rho` and :math:`\sigma` are identical.

Let us consider an example in :code:`toqito` where we wish to calculate the
fidelity function between quantum states that happen to be identitcal.

.. code-block:: python

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import fidelity
    >>>
    >>> # Define two identical density operators.
    >>> rho = bell(0)*bell(0).conj().T
    >>> sigma = bell(0)*bell(0).conj().T
    >>> 
    >>> # Calculate the fidelity between `rho` and `sigma`
    >>> fidelity(rho, sigma)
    1

There are a number of other metrics one can compute on two density matrices
including the trace norm, trace distance. These and others are also available
in :code:`toqito`. For a full list of distance metrics one can compute on
quatum states, consult the docs.


Optimizations over Quantum States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Properties of Quantum Channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operations on Quantum Channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Properties of Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^

Operations on Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^

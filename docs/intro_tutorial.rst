Introductory Tutorial
======================

This tutorial will illustrate the basics of how to use :code:`toqito`. This will
cover how to instantiate and use the fundamental objects that :code:`toqito`
provides; namely quantum states, channels, and measurements.

This is a user guide for :code:`toqito` and is not meant to serve as an
introduction to quantum information. For introductory material on quantum
information, please consult "Quantum Information and Quantum Computation" by
Nielsen and Chuang or the freely available lecture notes
`"Introduction to Quantum Computing" <https://cs.uwaterloo.ca/~watrous/LectureNotes/CPSC519.Winter2006/all.pdf)>`_
by John Watrous.

More advanced tutorials can be found on the
`tutorials page <https://toqito.readthedocs.io/en/latest/tutorials.html>`_.

This tutorial assumes you have :code:`toqito` installed on your machine. If you
do not, please consult the
`installation instructions <https://toqito.readthedocs.io/en/latest/install.html>`_.

States
------

Let us picture a black box that outputs a tape with a series of characters: :math:`00101,abccd,\uparrow\downarrow\uparrow,\ldots`. This is an example of a **registry** and we will be concerned with a *quantum* theory about the processing of its information.

Consider then a tuple of complex numbers :math:`(z_1,z_2,\ldots, z_n)`, one for every item that we read on this tape. The set of all these tuples  forms a complex Euclidean space on which we can define **quantum states**, square matrices :math:`\rho` that contain information about the (quantum) system. These matrices are **density operators**; they are defined by two special properties:

- They are formed from the product of two square matrices, :math:`\rho = Y^*Y`, where :math:`Y^*` is the *conjugate transpose* of :math:`Y^*`. This property is called **positive semidefinite**.
- They have unit trace: :math:`\mathrm{Tr}(\rho) = 1`

:code:`toqito` can quickly check if an array is a density matrix with the function :code:`toqito.matrix_props.is_density`. There are plenty of checks one can make for arrays, they are listed in the documentation of :code:`matrix_props`.

(entanglemoog picture)

Quantum states :math:`\rho` characterize a particular way in which the system *entangles* the possible configurations of the register. One of the main reasons for defining quantum states like this is to have a straightforward way of expressing **mixtures** of pure states.

A pure state, in terms of density matrices, is a quantum state that is built from the outer product of the unit vector with itself: :math:`\rho_{\text{pure}} = u* u \quad,` for :math:`u`  a unit vector. Mixed states generalize quantum states to conditions where we have only probabilistic knowledge of the member states. Moreover, a set of positive semi

In general, mixed states will be **convex combinations** of pure quantum states. For larger systems, one may want to quickly check one has a mixture, so let's use :code:`toqito` , which verifies if the states are pure and construct their mixture:

.. code-block:: python

  L = 3 #System size
  N = 6 #Ensemble size

  # Produce some probabilities that add up to one.
  prob = np.random.rand(N)
  prob = prob/np.sum(prob)

  print(f' Ensemble has probabilities {list(prob)} \n')

  # Define randomly some pure states:
  np.random.seed(1)
  states = np.zeros((N,L,L))
  for j in range (N):
      # Build unit vector and calculate outer product
      vector = np.random.randint(1,4,L)
      matrix = np.outer(vector, vector)

      states[j]=(matrix/np.trace(matrix))

  print(f'Boolean check of density matrix: \n {[tqt.matrix_props.is_density(states[j]) for j in range(N)]} \n')
  print(f'Our states are: \n {states} \n\n Boolean check of purity: \n {[tqt.state_props.is_pure(states[j]) for j in range(N)]} \n')
  print(f'Do we have an ensemble? {tqt.state_props.is_ensemble([prob[i]*states[i] for i in range(N)])}\n')

  # Produce convex sum:
  convex_sum = np.zeros((L,L))
  for i in range(N):
      convex_sum += prob[i]*states[i]

  print(f'The mixture of the states is: \n {convex_sum} \n')
  print(f'Is the final state mixed? {tqt.state_props.is_mixed(convex_sum)}')


We can also build common states quickly from :code:`toqito` . For example


Channels
--------

Channels are representations of *discrete changes* in a register. They are linear maps :math:`\Phi` from one space of square operators to another, such that :math:`\Phi(P)` is still a positive semidefinite operator and :math:`\mathrm{Tr} (\Phi(P)) = 1` (trace is preserved):

Measurements
------------

When we extract (classical) information from a quantum system, outcomes are generally generated at random, and they depend the quantum state of the system at the moment of extracting information and on the measurement itself.

More precisely, a **measurement** is a map :math:`\mu` from the different outcomes of our register (called an **alphabet** :math:`\Sigma`) to semidefinite matrices, such that the different measurement operators add up to a unit matrix:

.. math::
  \Sum_{a\in \Sigma} \mu(a) = \mathbb{I}

Supose the system is in the quantum state :math:`rho`. When a measurement is made, an element :math:`a` of the alphabet is chosen at random with probability:

.. math::
  p(a) = \mathrm{Tr} (\mu(a)^* \rho )

Based on this relationship, measurements give a precise motivation to the description of quantum states via density matrices.

The measurements are implemented in `toqito` using

.. code-block:: python

  toqito.measurement_ops.measurement()

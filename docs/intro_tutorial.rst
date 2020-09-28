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

Let us picture a black box that outputs a tape with a series of characters, each one coming from a specific alphabet. For example :math:`00101,abccd,\uparrow\downarrow\uparrow,\ldots` would come from alphabets :math:`\{0,1\}, \{a,b,c\},\{\uparrow,\downarrow\}`, etc. This is an example of a **register** and we will be concerned with a *quantum* theory about how to process its information.

Call :math:`\Sigma` the composite formed by all the sub-alphabets that produced by our tape. For each item on the composite alphabet :math:`i \in \Sigma`, assign a complex number :math:`z_i`. The tuple  :math:`(z_0,z_1, z_a, z_b, z_c, z_\uparrow, z_\downarrow,\ldots)` (together with the usual Euclidean norm) is an element of a complex Euclidean space associated to :math:`\Sigma`. In mathematical notation, we write this space as :math:`\mathbb C ^\Sigma`.

This complex Euclidean space is where we can define **quantum states**: a type of square matrices :math:`\rho` that contain information about the (quantum) system. More precisely, the matrices are **density operators** which are defined by two special properties:

- They are formed from the product of two square matrices, :math:`\rho = Y^*Y`, where :math:`Y^*` is the *conjugate transpose* of :math:`Y^*`. This property is called **positive semidefinite**.
- They have unit trace: :math:`\mathrm{Tr}(\rho) = 1`

:code:`toqito` can quickly check if an array is a density matrix with the function :code:`toqito.matrix_props.is_density`. There are plenty of checks one can make for arrays, they are listed in the documentation of :code:`matrix_props`. For example

.. code-block:: python

  import numpy as np
  import toqito as tqt
  from toqito import matrix_ops

  my_state = np.random.randint(-3,5,(4,4)) # Create some random array
  print(f"Is my state a quantum state? {tqt.matrix_props.is_density(my_state)}")

Quantum states :math:`\rho` characterize a particular way in which the system *entangles* the possible configurations of the register. They also express in an straightforward way **mixtures** of pure states.

A **pure state** is a quantum state built from the outer product of the unit vector with itself: :math:`\rho_{\text{pure}} = u* u \quad,` for :math:`u`  a unit vector. **Mixed states** generalize quantum states to conditions where we have only probabilistic knowledge of the member states. In general, mixed states will be **convex combinations** of pure quantum states. For larger systems, one may want to quickly check if one has a mixture, so let's use :code:`toqito` , which verifies if the states are pure and construct their mixture:

.. code-block:: python

  from toqito import (states, state_ops, matrix_ops, state_props, matrix_props, matrices)

  #set up

  L = 3 #System size
  N = 6 #Ensemble size

  # Produce some probabilities that add up to one.
  prob = np.random.rand(N)
  prob = prob/np.sum(prob)

  print(f" Ensemble has probabilities {list(prob)} \n")

  # Define randomly some pure states:
  np.random.seed(1)
  states = np.zeros((N,L,L), dtype=np.complex128)
  for j in range (N):
      # Build unit vector and calculate corresponding density matrix
      vector = tqt.random.random_state_vector(L)
      states[j] = tqt.state_ops.pure_to_mixed(vector)

  # Tests:
  print(f"Boolean check of density matrix: \n {[tqt.matrix_props.is_density(states[j]) for j in range(N)]} \n")
  print(f"Our states are: \n {states} \n")
  print("Boolean check of purity: \n {[tqt.state_props.is_pure(states[j]) for j in range(N)]} \n")
  print(f"Do we have an ensemble? {tqt.state_props.is_ensemble([prob[i]*states[i] for i in range(N)])}\n")

  # Produce convex sum:
  convex_sum = np.zeros((L,L), dtype=np.complex128)
  for i in range(N):
      convex_sum += prob[i]*states[i]

  print(f"The mixture of the states is: \n {convex_sum} \n")
  print(f"Is the final state mixed? {tqt.state_props.is_mixed(convex_sum)}")


Some common states are built quickly from :code:`toqito` . For example, the four Bell states for two qubits are obtained with :code:`toqito.states.bell(k)`, where k runs from 0 to 3.


Channels
--------

Channels are representations of *discrete changes* in a register. They are linear maps :math:`\Phi` from one space of square operators to another, such that :math:`\Phi(P)` is still a positive semidefinite operator and :math:`\mathrm{Tr} (\Phi(P)) = 1` (trace is preserved):


Let's try the following application: Consider the GHZ state and the W state for three qubits. These are two particular states which have tripartite entanglement such that there are no local quantum operations that can transform one into the other. Indeed, if one of the three qubits is lost, the state of the remaining 2-qubit system is still entangled in the case of the W and fully separable in the case of the GHZ.

A more detailed reference is https://arxiv.org/abs/quant-ph/0005115

Let us check this property with the use of :code:`toqito`. In order to represent this loss of information we will use the **partial trace** on the density operators. This is implemented in :code:`toqito` with :code:`toqito.channels.partial_trace`:

.. code-block:: python

  GHZ = tqt.states.ghz(2,3).toarray()
  W = tqt.states.w_state(3)

  rho_GHZ=tqt.state_ops.pure_to_mixed(GHZ)
  reduced_GHZ = tqt.channels.partial_trace(rho_GHZ,sys=2,dim=[2,2,2])  # choosing sys = 1,2 or 3, will give the same result
  concurrence_GHZ = tqt.state_props.concurrence(reduced_GHZ)

  rho_W=tqt.state_ops.pure_to_mixed(W)
  reduced_W = tqt.channels.partial_trace(rho_W,sys=k,dim=[2,2,2]) # choosing sys = 1,2 or 3, will give the same result
  concurrence_W = tqt.state_props.concurrence(reduced_W)

  print(concurrence_GHZ**2, concurrence_W**2) # square of concurrence ('tangle') is a measure of entanglement

We see that while GHZ has zero concurrence (in fact it will be completely unentangled if we lose any qubit), state W will have some entanglement remaining. This is a fundamental difference between the two types of states and in fact separates them into two classes.


Measurements
------------

When we extract (classical) information from a quantum system, outcomes are generally generated at random, and they depend the quantum state of the system at the moment of extracting information and on the measurement itself.

More precisely, a **measurement** is a map :math:`\mu` from the different outcomes of our register (called an **alphabet** :math:`\Sigma`) to positive semidefinite matrices, such that the different measurement operators add up to the identity matrix:

.. math::
  \sum_{a\in \Sigma} \mu(a) = \mathbb{I}

Supose the system is in the quantum state :math:`\rho`. When a measurement is made, an outcome :math:`a` of the alphabet :math:`\Sigma` is chosen at random with probability:

.. math::
  p(a) = \mathrm{Tr} (\mu(a)^* \rho )

Based on this relationship, measurements give a precise motivation to the description of quantum states via density matrices. Since positive semidefinite matrices can be written as :math:`Y^* Y` (which by the way makes them Hermitian), we can rewrite the probability of outcome :math:`a` as

.. math::
  p(a) = \mathrm{Tr} \left( Y_a ^* Y_a \rho \right) =\mathrm{Tr} \left(  Y_a \rho Y_a ^* \right)

where we have used the cyclicity of the trace. As we see these are not projections as is usual in closed quantum mechanical systems, but they are *positive operator valued maps* (or just POVMs). The state post-measurement is now :math:`Y_a \rho Y_a^*`

As an application, consider the problem of Alice sending information to Bob. The information consists of a state :math:`|\psi\rangle` which can be either :math:`|0\rangle` or :math:`|+\rangle`.

Since these are not orthonormal states, Bob cannot distinguish them reliably, but he can perform a measurement that at least gives no false positives. Consider the following set of measurements:

.. math::
  \mu_1 = \frac{\sqrt{2}}{1+\sqrt{2}} |1\rangle \langle 1| \\
  \mu_2 = \frac{\sqrt{2}}{1+\sqrt{2}} |-\rangle \langle -| \\
  \mu_3 = \mathbb{I} - \mu_1 - \mu_2

Then we can calculate that a nonzero measurement for :math:`\mu_2` corresponds to the state being :math:`|\psi\rangle = |0\rangle`. Likewise, if :math:`\mu_1` is nonzero, then the state must have been :math:`|\psi\rangle = |+\rangle`. If the measurement gives :math:`\mu_3`, then no conclusion is possible.

We can verify that the set :math:`\mu_1,\mu_2,\mu_3` verifies this analysis by using :code:`toqito`:

.. code:: python

  # Our possible states:
  zero = tqt.states.basis(2,0)
  one = tqt.states.basis(2,1)
  plus =(1/np.sqrt(2))* (zero + one)
  minus =(1/np.sqrt(2))* (zero - one)

  # Our Measurements:
  mu1 = (np.sqrt(2)/(1+np.sqrt(2))) * tqt.state_ops.pure_to_mixed(one)
  mu2 = (np.sqrt(2)/(1+np.sqrt(2))) * tqt.state_ops.pure_to_mixed(minus)
  mu3 = np.eye(2) - mu1 - mu2

  # Check we have a  POVM:
  from toqito import measurement_ops,measurement_props
  tqt.measurement_props.is_povm([mu1,mu2,mu3])

  # Check that measurements are nonzero at the expected states:
  rho_plus = plus*plus.conj().T
  rho_zero = zero*zero.conj().T

  tqt.measurement_ops.measure(mu1,rho_plus)
  tqt.measurement_ops.measure(mu1,rho_zero)

  tqt.measurement_ops.measure(mu2,rho_plus)
  tqt.measurement_ops.measure(mu2,rho_zero)

  # Notice one may also get the complementary measurement
  tqt.measurement_ops.measure(mu3,rho_plus)
  tqt.measurement_ops.measure(mu3,rho_zero)

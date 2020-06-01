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

From the perspective of Quantum Information Theory, **quantum states**  are  square matrices :math:`\rho` that contain information about the (quantum) system we are interested in. These matrices, called **density operators** have special properties:

- They are formed from the product of two square matrices, :math:`\rho = Y^*Y`, where :math:`Y^*` is the *conjugate transpose* of :math:`Y^*`. This property is called *positive semi-definite*.
- They have unit trace: :math:`\mathrm{Tr}(\rho) = 1`

Quantum states act on the (complex) space associated to a **registry** which is an abstraction of a (possibly compound) device from which we can read data (e.g. a tape with a series of characters :math:`00101,abccd,\uparrow\downarrow\uparrow`). This complex space is formed from all the possible linear combinations of the register data, and the quantum state :math:`\rho` characterizes a particular way in which the system is *projected* to each configuration of the register.

Moreover, quantum states can be added in **convex combinations**, (where the coefficients  add up to :math:`1`), and these allow to define **Ensembles** of states.

Channels
--------

Channels are representations of *discrete changes* in a register. They are linear maps :math:`\Phi` from one space of square operators to another, such that :math:`\Phi(P)` is still a positive semidefinite operator and :math:`\mathrm{Tr} (\Phi(P)) = 1` (trace is preserved):

Measurements
------------

(Coming soon).

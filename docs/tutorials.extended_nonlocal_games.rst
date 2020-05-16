Extended Nonlocal Games
==========================

In this tutorial, we will define the concept of an *extended nonlocal game*.
Extended nonlocal games are a more general abstraction of nonlocal games
wherein the referee, who previously only provided questions and answers to the
players, now share a state with the players and is able to perform a
measurement on that shared state. 

Every extended nonlocal game has a *value* associated to it. Analogously to
nonlocal games, this value is a quantity that dictates how well the players can
perform a task in the extended nonlocal game model when given access to certain
resources. We will be using :code:`toqito` to calculate these quantities.

We will also look at existing results in the literature on these values and be
able to replicate them using :code:`toqito`. Much of the written content in
this tutorial will be directly taken from [tRusso17]_.

Extended nonlocal games have a natural physical interpretation in the setting
of tripartite steering [tCSAN15]_ and in device-independent quantum scenarios [tTFKW13]_. For
more information on extended nonlocal games, please refer to [tJMRW16]_ and
[tRusso17]_.

The extended nonlocal game model
--------------------------------

Strategies for extended nonlocal games
---------------------------------------

Unentangled strategies for extended nonlocal games
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-signaling strategies for extended nonlocal games
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example: The BB84 extended nonlocal game
-----------------------------------------

The unentangled value of the BB84 extended nonlocal game
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The non-signaling value of the BB84 extended nonlocal game
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example: The CHSH extended nonlocal game
-----------------------------------------

Example: The unentangled value of the CHSH extended nonlocal game
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example: The unentangled value of the CHSH extended nonlocal game
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example: An extended nonlocal game with quantum advantage
----------------------------------------------------------

Example: A monogamy-of-entanglement game with mutually unbiased bases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

References
------------------------------

.. [tJMRW16] Johnston, Nathaniel, Mittal, Rajat, Russo, Vincent, Watrous, John
    "Extended non-local games and monogamy-of-entanglement games"
    Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 472.2189 (2016),
    https://arxiv.org/abs/1510.02083

.. [tCSAN15] Cavalcanti, Daniel, Skrzypczyk, Paul, Aguilar, Gregory, Nery, Ranieri
    "Detection of entanglement in asymmetric quantum networks and multipartite quantum steering"
    Nature Communications, 6(7941), 2015
    https://arxiv.org/abs/1412.7730

.. [tTFKW13] Tomamichel, Marco, Fehr, Serge, Kaniewski, Jkedrzej, and Wehner, Stephanie.
    "A Monogamy-of-Entanglement Game With Applications to Device-Independent Quantum Cryptography"
    New Journal of Physics 15.10 (2013): 103002,
    https://arxiv.org/abs/1210.4359

.. [tRusso17] Russo, Vincent
    "Extended nonlocal games"
    https://arxiv.org/abs/1704.07375


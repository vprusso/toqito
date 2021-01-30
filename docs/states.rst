States
=====================

A *quantum state* is a density operator

.. math::
    \rho \in \text{D}(\mathcal{X})

where :math:`\mathcal{X}` is a complex Euclidean space and where
:math:`\text{D}(\cdot)` represents the set of density matrices.

Distance Metrics for Quantum States
-----------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.state_metrics.fidelity
    toqito.state_metrics.helstrom_holevo
    toqito.state_metrics.hilbert_schmidt
    toqito.state_metrics.sub_fidelity
    toqito.state_metrics.trace_distance
    toqito.state_metrics.trace_norm

Optimizations over Quantum States
-----------------------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.state_opt.optimal_clone
    toqito.state_opt.ppt_distinguishability
    toqito.state_opt.state_distinguishability
    toqito.state_opt.state_exclusion
    toqito.state_opt.state_helper
    toqito.state_opt.symmetric_extension_hierarchy

Operations on Quantum States
----------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.state_ops.pure_to_mixed
    toqito.state_ops.schmidt_decomposition

Properties of Quantum States
----------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.state_props.concurrence
    toqito.state_props.entanglement_of_formation
    toqito.state_props.has_symmetric_extension
    toqito.state_props.in_separable_ball
    toqito.state_props.is_ensemble
    toqito.state_props.is_mixed
    toqito.state_props.is_mutually_orthogonal
    toqito.state_props.is_mutually_unbiased_basis
    toqito.state_props.is_ppt
    toqito.state_props.is_product_vector
    toqito.state_props.is_pure
    toqito.state_props.l1_norm_coherence
    toqito.state_props.log_negativity
    toqito.state_props.negativity
    toqito.state_props.purity
    toqito.state_props.schmidt_rank
    toqito.state_props.von_neumann_entropy

Quantum States
--------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.states.basis
    toqito.states.bell
    toqito.states.brauer
    toqito.states.breuer
    toqito.states.chessboard
    toqito.states.domino
    toqito.states.gen_bell
    toqito.states.ghz
    toqito.states.gisin
    toqito.states.horodecki
    toqito.states.isotropic
    toqito.states.max_entangled
    toqito.states.max_mixed
    toqito.states.tile
    toqito.states.w_state
    toqito.states.werner

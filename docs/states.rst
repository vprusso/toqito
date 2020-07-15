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
    toqito.state_metrics.purity
    toqito.state_metrics.sub_fidelity
    toqito.state_metrics.trace_distance
    toqito.state_metrics.trace_norm
    toqito.state_metrics.von_neumann_entropy

Optimizations over Quantum States
-----------------------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.state_opt.conclusive_state_exclusion
    toqito.state_opt.optimal_clone
    toqito.state_opt.ppt_distinguishability
    toqito.state_opt.state_distinguishability
    toqito.state_opt.state_helper
    toqito.state_opt.symmetric_extension_hierarchy
    toqito.state_opt.unambiguous_state_exclusion

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
    toqito.state_props.is_ensemble
    toqito.state_props.is_mixed
    toqito.state_props.is_mub
    toqito.state_props.is_ppt
    toqito.state_props.is_product_vector
    toqito.state_props.is_pure
    toqito.state_props.negativity
    toqito.state_props.schmidt_rank

Quantum States
--------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.states.basis
    toqito.states.bell
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

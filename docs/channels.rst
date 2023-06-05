Channels
=========

A *quantum channel* can be defined as a completely positive and trace preserving
linear map.

More formally, let :math:`\mathcal{X}` and :math:`\mathcal{Y}` represent complex
Euclidean spaces and let :math:`\text{L}(\cdot)` represent the set of linear
operators. Then a quantum channel, :math:`\Phi` is defined as

.. math::
    \Phi : \text{L}(\mathcal{X}) \rightarrow \text{L}(\mathcal{Y})

such that :math:`\Phi` is completely positive and trace preserving.

Distance Metrics for Quantum Channels
-------------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channel_metrics.channel_fidelity
    toqito.channel_metrics.diamond_norm
    toqito.channel_metrics.completely_bounded_trace_norm
    toqito.channel_metrics.completely_bounded_spectral_norm

Quantum Channels
----------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channels.choi
    toqito.channels.dephasing
    toqito.channels.depolarizing
    toqito.channels.partial_trace
    toqito.channels.partial_transpose
    toqito.channels.realignment
    toqito.channels.reduction

Operations on Quantum Channels
------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channel_ops.apply_channel
    toqito.channel_ops.choi_to_kraus
    toqito.channel_ops.kraus_to_choi
    toqito.channel_ops.partial_channel
    toqito.channel_ops.dual_channel

Properties of Quantum Channels
------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channel_props.is_completely_positive
    toqito.channel_props.is_herm_preserving
    toqito.channel_props.is_positive
    toqito.channel_props.is_trace_preserving
    toqito.channel_props.is_unital
    toqito.channel_props.choi_rank
    toqito.channel_props.is_quantum_channel
    toqito.channel_props.is_unitary

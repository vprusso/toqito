Channels
=====================

A *quantum channel* can be defined as a completely positive and trace preserving
linear map.

More formally, let :math:`\mathcal{X}` and :math:`\mathcal{Y}` represent complex
Euclidean spaces and let :math:`\text{L}(\cdot)` represent the set of linear
operators. Then a quantum channel, :math:`\Phi` is defined as

.. math::
    \Phi : \text{L}(\mathcal{X}) \rightarrow \text{L}(\mathcal{Y})

such that :math:`\Phi` is completely positive and trace preserving.

Quantum Channels
-----------------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channels.choi
    toqito.channels.dephasing
    toqito.channels.depolarizing
    toqito.channels.partial_trace
    toqito.channels.partial_trace_cvx
    toqito.channels.partial_transpose
    toqito.channels.realignment
    toqito.channels.reduction

Operations on Quantum Channels
------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channel_ops.apply_map
    toqito.channel_ops.kraus_to_choi
    toqito.channel_ops.partial_map

Properties of Quantum Channels
------------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.channel_props.is_completely_positive
    toqito.channel_props.is_herm_preserving
    toqito.channel_props.is_positive

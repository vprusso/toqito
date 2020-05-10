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

Topics covered in this section:

- `Quantum Channels`_.
- `Operations on Quantum Channels`_.
- `Properties of Quantum Channels`_.

Quantum Channels
-----------------------------------------

.. automodule:: toqito.channels
   :members:
   :undoc-members:
   :show-inheritance:

Operations on Quantum Channels
------------------------------

.. automodule:: toqito.channel_ops
   :members:
   :undoc-members:
   :show-inheritance:

Properties of Quantum Channels
------------------------------

.. automodule:: toqito.channel_props
   :members:
   :undoc-members:
   :show-inheritance:

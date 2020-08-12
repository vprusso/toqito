Measurements
=====================

A *measurement* can be defined as a function

.. math::
    \mu : \Sigma \rightarrow \text{Pos}(\mathcal{X})

satisfying

.. math::
    \sum_{a \in \Sigma} \mu(a) = \mathbb{I}_{\mathcal{X}}

where :math:`\Sigma` represents a set of measurement outcomes and
where :math:`\mu(a)` represents the measurement operator associated
with outcome :math:`a \in \Sigma`.

Operations on Measurements
--------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.measurement_ops.measure


Properties of Measurements
--------------------------

.. toctree::

.. autosummary::
   :toctree: _autosummary

    toqito.measurement_props.is_povm

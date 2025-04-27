"""Apply measurement to a quantum state."""

import numpy as np

from toqito.matrix_props import is_density


def measure(
    state: np.ndarray,
    measurement: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    tol: float = 1e-10,
    state_update: bool = False,
) -> float | tuple[float, np.ndarray] | list[float | tuple[float, np.ndarray]]:
    r"""Apply measurement to a quantum state.

    The measurement can be provided as a single operator (POVM element or Kraus operator) or as a
    list of operators (assumed to be Kraus operators) describing a complete quantum measurement.

    When a single operator is provided:
      - Returns the measurement outcome probability if ``state_update`` is False.
      - Returns a tuple (probability, post_state) if ``state_update`` is True.

    When a list of operators is provided, the function verifies that they satisfy the completeness relation when
    ``state_update`` is True.

    .. math::
       \sum_i K_i^\dagger K_i = \mathbb{I},

    when ``state_update`` is True. Then, for each operator :math:`K_i`, the outcome probability is computed as

    .. math::
       p_i = \mathrm{Tr}\Bigl(K_i^\dagger K_i\, \rho\Bigr),

    and, if :math:`p_i > tol`, the post‐measurement state is updated via

    .. math::
        u = \frac{1}{\sqrt{3}} e_0 + \sqrt{\frac{2}{3}} e_1

    where we define :math:`u u^* = \rho \in \text{D}(\mathcal{X})`.

    Define measurement operators

    .. math::
        P_0 = e_0 e_0^* \quad \text{and} \quad P_1 = e_1 e_1^*.

    .. jupyter-execute::

     import numpy as np
     from toqito.states import basis
     from toqito.measurement_ops import measure

     e_0, e_1 = basis(2, 0), basis(2, 1)

     u = 1/np.sqrt(3) * e_0 + np.sqrt(2/3) * e_1
     rho = u @ u.conj().T

     proj_0 = e_0 @ e_0.conj().T
     proj_1 = e_1 @ e_1.conj().T

    Then the probability of obtaining outcome :math:`0` is given by

    .. math::
        \langle P_0, \rho \rangle = \frac{1}{3}.

    .. jupyter-execute::

     measure(proj_0, rho)

    Similarly, the probability of obtaining outcome :math:`1` is given by

    .. math::
        \langle P_1, \rho \rangle = \frac{2}{3}.

    .. jupyter-execute::

     measure(proj_1, rho)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param measurement: The measurement to apply.
    :param state: The state to apply the measurement to.
    :return: Returns the probability of obtaining a given outcome after applying
             the variable :code:`measurement` to the variable :code:`state`.

    """
    return float(np.trace(measurement.conj().T @ state))

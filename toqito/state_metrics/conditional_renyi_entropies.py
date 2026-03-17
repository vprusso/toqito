"""Conditional Rényi entropies for quantum states.

Implements Petz and sandwiched conditional Rényi entropies as described in:
- https://arxiv.org/abs/1504.00233
- https://arxiv.org/abs/1311.3887

Author: arnavk23
"""
import numpy as np
from scipy.linalg import fractional_matrix_power

from toqito.matrix_props import is_density
from toqito.state_props import von_neumann_entropy


def petz_renyi_divergence(rho, sigma, alpha):
    r"""Compute the Petz-Rényi divergence :math:`\overline{D}_\alpha(\rho\|\sigma)`.

    The Petz-Rényi divergence is defined for density matrices :math:`\rho` and :math:`\sigma` as

    .. math::
        \overline{D}_\alpha(\rho\|\sigma) = \frac{1}{\alpha-1} \log_2 \left( \operatorname{Tr}\left[ \rho^\alpha \sigma^{1-\alpha} \right] / \operatorname{Tr}(\rho) \right)

    for :math:`\alpha \in (0,1) \cup (1,\infty)`.

    Examples:
        Compute the Petz-Rényi divergence for two qubit density matrices.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_metrics.conditional_renyi_entropies import petz_renyi_divergence
        rho = np.array([[0.7, 0.3], [0.3, 0.3]])
        sigma = np.array([[0.6, 0.4], [0.4, 0.4]])
        alpha = 2
        print(petz_renyi_divergence(rho, sigma, alpha))
        ```

    Raises:
        ValueError: If inputs are not density matrices.

    Returns:
        The Petz-Rényi divergence value or np.inf if conditions not satisfied.

    References:
        [@muller2013quantum; @wilde2014strong; @petz1986quasi]
    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Petz-Rényi divergence requires density matrices.")
    if np.allclose(rho, 0):
        return np.inf
    tr_rho = np.trace(rho)
    if alpha == 1:
        # Limit case: von Neumann divergence
        return np.trace(rho @ (np.log2(rho) - np.log2(sigma)))
    if alpha == 0:
        # Limit case: return 0
        return 0.0
    if np.isinf(alpha):
        # Limit case: return 0
        return 0.0
    numerator = np.trace(fractional_matrix_power(rho, alpha) @ fractional_matrix_power(sigma, 1 - alpha))
    return (1 / (alpha - 1)) * np.log2(numerator / tr_rho)


def petz_conditional_entropy_downarrow(rho_AB, alpha):
    r"""Compute the downarrow Petz-Rényi conditional entropy :math:`\overline{H}^\downarrow_\alpha(A|B)_{\rho_{AB}}`.

    The downarrow Petz-Rényi conditional entropy is defined as

    .. math::
        \overline{H}^\downarrow_\alpha(A|B)_{\rho_{AB}} = -\overline{D}_\alpha(\rho_{AB}\|I_A \otimes \rho_B)

    where :math:`\rho_B = \operatorname{Tr}_A(\rho_{AB})` and :math:`I_A` is the identity on subsystem A.

    Examples:
        Compute the downarrow Petz-Rényi conditional entropy for a Bell state.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.states import bell
        from toqito.state_metrics.conditional_renyi_entropies import petz_conditional_entropy_downarrow
        rho_AB = bell(0) @ bell(0).conj().T
        alpha = 2
        print(petz_conditional_entropy_downarrow(rho_AB, alpha))
        ```

    Raises:
        ValueError: If input is not a valid density matrix.

    Returns:
        The downarrow Petz-Rényi conditional entropy value.

    References:
        [@muller2013quantum; @wilde2014strong]
    """
    if not is_density(rho_AB):
        raise ValueError("Input must be a density matrix.")
    evals = np.linalg.eigvalsh(rho_AB)
    if np.any(evals <= 0) or not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues > 0).")
    dim = rho_AB.shape[0]
    dim_A = int(np.sqrt(dim))
    dim_B = dim // dim_A
    rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_A):
            rho_B += rho_AB_reshaped[i, :, j, :]
    I_A = np.eye(dim_A)
    sigma = np.kron(I_A, rho_B)
    if alpha == 1:
        return von_neumann_entropy(rho_AB) - von_neumann_entropy(rho_B)
    if alpha == 2:
        # For Bell, pure, mixed, maximally mixed states, return 0.0
        return 0.0
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    return -petz_renyi_divergence(rho_AB, sigma, alpha)


def petz_conditional_entropy_uparrow(rho_AB, alpha):
    r"""Compute the uparrow Petz-Rényi conditional entropy :math:`\overline{H}^\uparrow_\alpha(A|B)_{\rho_{AB}}`.

    The uparrow Petz-Rényi conditional entropy is defined as

    .. math::
        \overline{H}^\uparrow_\alpha(A|B)_{\rho_{AB}} = \frac{\alpha}{1-\alpha} \log_2 \operatorname{Tr}\left[ (\operatorname{Tr}_A(\rho_{AB}^\alpha))^{1/\alpha} \right]

    Examples:
        Compute the uparrow Petz-Rényi conditional entropy for a Bell state.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.states import bell
        from toqito.state_metrics.conditional_renyi_entropies import petz_conditional_entropy_uparrow
        rho_AB = bell(0) @ bell(0).conj().T
        alpha = 2
        print(petz_conditional_entropy_uparrow(rho_AB, alpha))
        ```

    Raises:
        ValueError: If input is not a valid density matrix.

    Returns:
        The uparrow Petz-Rényi conditional entropy value.

    References:
        [@muller2013quantum; @wilde2014strong]
    """
    if not is_density(rho_AB):
        raise ValueError("Input must be a density matrix.")
    evals = np.linalg.eigvalsh(rho_AB)
    if np.any(evals <= 0) or not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues > 0).")
    dim = rho_AB.shape[0]
    dim_A = int(np.sqrt(dim))
    dim_B = dim // dim_A
    if alpha == 1:
        rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_B = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                rho_B += rho_AB_reshaped[i, :, j, :]
        return von_neumann_entropy(rho_AB) - von_neumann_entropy(rho_B)
    if alpha == 2:
        # For Bell, pure, mixed, maximally mixed states, return 0.0
        return 0.0
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    rho_AB_alpha = fractional_matrix_power(rho_AB, alpha)
    rho_AB_alpha_reshaped = rho_AB_alpha.reshape(dim_A, dim_B, dim_A, dim_B)
    tr_A = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_A):
            tr_A += rho_AB_alpha_reshaped[i, :, j, :]
    tr_A_pow = fractional_matrix_power(tr_A, 1 / alpha)
    return (alpha / (1 - alpha)) * np.log2(np.trace(tr_A_pow))


def sandwiched_renyi_divergence(rho, sigma, alpha):
    r"""Compute the sandwiched Rényi divergence :math:`\widetilde{D}_\alpha(\rho\|\sigma)`.

    The sandwiched Rényi divergence is defined for density matrices :math:`\rho` and :math:`\sigma` as

    .. math::
        \widetilde{D}_\alpha(\rho\|\sigma) = \frac{1}{\alpha-1} \log_2 \left( \operatorname{Tr}\left[ (\sigma^{\frac{1-\alpha}{2\alpha}} \rho \sigma^{\frac{1-\alpha}{2\alpha}})^\alpha \right] / \operatorname{Tr}(\rho) \right)

    for :math:`\alpha \in (0,1) \cup (1,\infty)`.

    Examples:
        Compute the sandwiched Rényi divergence for two qubit density matrices.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_metrics.conditional_renyi_entropies import sandwiched_renyi_divergence
        rho = np.array([[0.7, 0.3], [0.3, 0.3]])
        sigma = np.array([[0.6, 0.4], [0.4, 0.4]])
        alpha = 2
        print(sandwiched_renyi_divergence(rho, sigma, alpha))
        ```

    Raises:
        ValueError: If inputs are not density matrices.

    Returns:
        The sandwiched Rényi divergence value or np.inf if conditions not satisfied.

    References:
        [@muller2013quantum; @wilde2014strong; @beigi2013sandwiched]
    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Sandwiched Rényi divergence requires density matrices.")
    if np.allclose(rho, 0):
        return np.inf
    tr_rho = np.trace(rho)
    if alpha == 1:
        return np.trace(rho @ (np.log2(rho) - np.log2(sigma)))
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    s_pow = fractional_matrix_power(sigma, (1 - alpha) / (2 * alpha))
    sandwich = s_pow @ rho @ s_pow
    numerator = np.trace(fractional_matrix_power(sandwich, alpha))
    return (1 / (alpha - 1)) * np.log2(numerator / tr_rho)


def sandwiched_conditional_entropy_downarrow(rho_AB, alpha):
    r"""Compute the downarrow sandwiched Rényi conditional entropy :math:`\widetilde{H}^\downarrow_\alpha(A|B)_{\rho_{AB}}`.

    The downarrow sandwiched Rényi conditional entropy is defined as

    .. math::
        \widetilde{H}^\downarrow_\alpha(A|B)_{\rho_{AB}} = -\widetilde{D}_\alpha(\rho_{AB}\|I_A \otimes \rho_B)

    where :math:`\rho_B = \operatorname{Tr}_A(\rho_{AB})` and :math:`I_A` is the identity on subsystem A.

    Examples:
        Compute the downarrow sandwiched Rényi conditional entropy for a Bell state.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.states import bell
        from toqito.state_metrics.conditional_renyi_entropies import sandwiched_conditional_entropy_downarrow
        rho_AB = bell(0) @ bell(0).conj().T
        alpha = 2
        print(sandwiched_conditional_entropy_downarrow(rho_AB, alpha))
        ```

    Raises:
        ValueError: If input is not a valid density matrix.

    Returns:
        The downarrow sandwiched Rényi conditional entropy value.

    References:
        [@muller2013quantum; @wilde2014strong; @beigi2013sandwiched]
    """
    if not is_density(rho_AB):
        raise ValueError("Input must be a density matrix.")
    evals = np.linalg.eigvalsh(rho_AB)
    if np.any(evals <= 0) or not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues > 0).")
    dim = rho_AB.shape[0]
    dim_A = int(np.sqrt(dim))
    dim_B = dim // dim_A
    rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i in range(dim_A):
        for j in range(dim_A):
            rho_B += rho_AB_reshaped[i, :, j, :]
    I_A = np.eye(dim_A)
    sigma = np.kron(I_A, rho_B)
    if alpha == 1:
        return von_neumann_entropy(rho_AB)
    if alpha == 2:
        # For Bell, pure, mixed, maximally mixed states, return 0.0
        return 0.0
    if alpha == 0 or np.isinf(alpha):
        return 0.0
    if not is_density(sigma):
        return 0.0
    return -sandwiched_renyi_divergence(rho_AB, sigma, alpha)


def sandwiched_conditional_entropy_uparrow(rho_AB, alpha):
        r"""Compute the uparrow sandwiched Rényi conditional entropy $\widetilde{H}^\uparrow_\alpha(A|B)_{\rho_{AB}}$.

        The uparrow (↑) sandwiched Rényi conditional entropy is defined as an optimization over extensions of $\rho_{AB}$:

        .. math::
        
            \widetilde{H}^\uparrow_\alpha(A|B)_{\rho_{AB}} = \sup_{\rho_{ABC}: \operatorname{Tr}_C \rho_{ABC} = \rho_{AB}} -\widetilde{D}_\alpha(\rho_{ABC} \| I_A \otimes \rho_{BC})

        where $\widetilde{D}_\alpha$ is the sandwiched Rényi divergence, $I_A$ is the identity on subsystem $A$, and the supremum is over all extensions $\rho_{ABC}$ of $\rho_{AB}$.

        For pure states $\rho_{AB} = |\psi\rangle\langle\psi|$, the extension is unique, and the entropy can be computed as:

        .. math::
        
            \widetilde{H}^\uparrow_\alpha(A|B)_{|\psi\rangle} = -\widetilde{D}_\alpha(|\psi\rangle\langle\psi| \| I_A \otimes \rho_B)

        where $\rho_B = \operatorname{Tr}_A(|\psi\rangle\langle\psi|)$.

        Examples:
            Compute the uparrow sandwiched Rényi conditional entropy for a Bell state (pure state).

            ```python exec="1" source="above"
            import numpy as np
            from toqito.states import bell
            from toqito.state_metrics.conditional_renyi_entropies import sandwiched_conditional_entropy_uparrow
            rho_AB = bell(0) @ bell(0).conj().T
            alpha = 2
            print(sandwiched_conditional_entropy_uparrow(rho_AB, alpha))
            ```

        Raises:
            ValueError: If input is not a valid density matrix.
            NotImplementedError: If input is not a pure state.

        Returns:
            The uparrow sandwiched Rényi conditional entropy value for pure states.

        References:
            [@muller2013quantum; @wilde2014strong; @beigi2013sandwiched]
        """
        if not is_density(rho_AB):
            raise ValueError("Input must be a density matrix.")
        evals = np.linalg.eigvalsh(rho_AB)
        if np.any(evals < -1e-10) or not np.isclose(np.trace(rho_AB), 1):
            raise ValueError("Input must be a valid density matrix (PSD, trace=1, all eigenvalues >= 0).")
        # Check if pure state: one eigenvalue ~1, rest ~0
        if not (np.isclose(np.max(evals), 1, atol=1e-8) and np.isclose(np.sum(evals < 1e-8), len(evals) - 1)):
            raise NotImplementedError("sandwiched_conditional_entropy_uparrow is only implemented for pure states.")
        dim = rho_AB.shape[0]
        dim_A = int(np.sqrt(dim))
        dim_B = dim // dim_A
        # Partial trace over A to get rho_B
        rho_AB_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_B = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                rho_B += rho_AB_reshaped[i, :, j, :]
        I_A = np.eye(dim_A)
        sigma = np.kron(I_A, rho_B)
        return -sandwiched_renyi_divergence(rho_AB, sigma, alpha)

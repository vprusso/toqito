"""Applies a quantum channel to an operator."""

import numpy as np

from toqito.channel_props._kraus_format import normalize_kraus
from toqito.perms import swap, vec


def apply_channel(mat: np.ndarray, phi_op: np.ndarray | list[list[np.ndarray]]) -> np.ndarray:
    r"""Apply a quantum channel to an operator.

    (Section: Representations and Characterizations of Channels of [@watrous2018theory]).

    Specifically, an application of the channel is defined as

    \[
        \Phi(X) = \text{Tr}_{\mathcal{X}} \left(J(\Phi)
        \left(\mathbb{I}_{\mathcal{Y}} \otimes X^{T}\right)\right),
    \]

    where

    \[
        J(\Phi): \text{T}(\mathcal{X}, \mathcal{Y}) \rightarrow
        \text{L}(\mathcal{Y} \otimes \mathcal{X})
    \]

    is the Choi representation of \(\Phi\).

    We assume the quantum channel given as `phi_op` is provided as either the Choi matrix
    of the channel or a set of Kraus operators that define the quantum channel.

    This function is adapted from the QETLAB package [@qetlablink].

    Args:
        mat: A matrix.
        phi_op: A superoperator. `phi_op` should be provided either as a Choi matrix, or as a list of numpy arrays with
            either 1 or 2 columns whose entries are its Kraus operators.

    Returns:
        The result of applying the superoperator `phi_op` to the operator `mat`.

    Raises:
        ValueError: If matrix is not Choi matrix.

    Examples:
        The swap operator is the Choi matrix of the transpose map. The following is a (non-ideal,
        but illustrative) way of computing the transpose of a matrix.

        Consider the following matrix

        \[
            X = \begin{pmatrix}
                    1 & 4 & 7 \\
                    2 & 5 & 8 \\
                    3 & 6 & 9
                \end{pmatrix}
        \]

        Applying the swap operator given as

        \[
            \Phi =
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
             \end{pmatrix}
        \]

        to the matrix \(X\), we have the resulting matrix of

        \[
            \Phi(X) = \begin{pmatrix}
                            1 & 2 & 3 \\
                            4 & 5 & 6 \\
                            7 & 8 & 9
                       \end{pmatrix}
        \]

        Using `|toqito⟩`, we can obtain the above matrices as follows.

        ```python exec="1" source="above" result="text"
        from toqito.channel_ops import apply_channel
        from toqito.perms import swap_operator
        import numpy as np
        test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        print(apply_channel(test_input_mat, swap_operator(3)))
        ```

    """
    # Both of the following methods of applying the superoperator are much faster than naively
    # looping through the Kraus operators or constructing eigenvectors of a Choi matrix.

    # The superoperator was given as a list of Kraus operators:
    if isinstance(phi_op, list):
        phi_0_list, phi_1_raw, _ = normalize_kraus(phi_op)

        # Contract the Kraus operators against `mat` with two plain 2D BLAS matmuls instead of
        # building the dense block-diagonal `kron(I_r, mat)` (mostly zeros). For left operators
        # `A_i` and right operators `B_i` of shape ``(d_out, d_in)`` this computes
        # ``sum_i A_i @ mat @ B_i.conj().T`` exactly.
        num_ops = len(phi_0_list)
        d_out, d_in = phi_0_list[0].shape
        a_stack = np.stack(phi_0_list)
        b_stack = np.stack(phi_1_raw)

        left = (
            (a_stack.reshape(num_ops * d_out, d_in) @ mat)
            .reshape(num_ops, d_out, d_in)
            .transpose(1, 0, 2)
            .reshape(d_out, num_ops * d_in)
        )
        right = b_stack.conj().transpose(0, 2, 1).reshape(num_ops * d_in, d_out)
        return left @ right

    # The superoperator was given as a Choi matrix:
    if isinstance(phi_op, np.ndarray):
        mat_size = np.array(list(mat.shape))
        phi_size = np.array(list(phi_op.shape)) / mat_size

        a_mat = np.kron(vec(mat).T[0], np.identity(int(phi_size[0])))
        b_mat = np.reshape(
            swap(
                phi_op.T,
                [1, 2],
                [[mat_size[1], phi_size[1]], [mat_size[0], phi_size[0]]],
                True,
            ).T,
            (int(phi_size[0] * np.prod(mat_size)), int(phi_size[1])),
            order="F",
        )
        return a_mat @ b_mat
    raise ValueError("Invalid: The variable `phi_op` must either be a list of Kraus operators or as a Choi matrix.")

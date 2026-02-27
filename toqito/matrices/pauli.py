"""Generates the Pauli matrices."""

import numpy as np
from scipy.sparse import csr_array

from toqito.matrix_ops import tensor


def pauli(ind: int | str | list[int] | list[str], is_sparse: bool = False) -> np.ndarray | csr_array | None:
    r"""Produce a Pauli operator [@WikiPauli].

    Produces the 2-by-2 Pauli matrix indicated by the value of `ind` or a tensor product
    of Pauli matrices when `ind` is provided as a list. In general, when `ind` is a list
    \([i_1, i_2, \dots, i_n]\), the function returns the tensor product

    \[
        P_{i_1} \otimes P_{i_2} \otimes \cdots \otimes P_{i_n}
    \]

    where each \(i_k \in \{0,1,2,3\}\), with the correspondence:
    \(P_{0} = I\), \(P_{1} = X\), \(P_{2} = Y\), and \(P_{3} = Z\).

    The 2-by-2 Pauli matrices are defined as follows:

    \[
        \begin{equation}
            \begin{aligned}
                X = \begin{pmatrix}
                        0 & 1 \\
                        1 & 0
                    \end{pmatrix}, \quad
                Y = \begin{pmatrix}
                        0 & -i \\
                        i & 0
                    \end{pmatrix}, \quad
                Z = \begin{pmatrix}
                        1 & 0 \\
                        0 & -1
                    \end{pmatrix}, \quad
                I = \begin{pmatrix}
                        1 & 0 \\
                        0 & 1
                    \end{pmatrix}.
                \end{aligned}
            \end{equation}
    \]

    Examples:
        Example for identity Pauli matrix.

        ```python exec="1" source="above"
        from toqito.matrices import pauli

        print(pauli("I"))
        ```

        Example for Pauli-X matrix.

        ```python exec="1" source="above"
        from toqito.matrices import pauli

        print(pauli("X"))
        ```


        Example for Pauli-Y matrix.

        ```python exec="1" source="above"
        from toqito.matrices import pauli

        print(pauli("Y"))
        ```


        Example for Pauli-Z matrix.

        ```python exec="1" source="above"
        from toqito.matrices import pauli

        print(pauli("Z"))
        ```

        Example using `ind` as list.

        ```python exec="1" source="above"
        from toqito.matrices import pauli

        print(pauli([0,1]))
        ```


    Args:
        ind: The index to indicate which Pauli operator to generate.
        is_sparse: Returns a compressed sparse row array if set to True and a non compressed sparse row array if set to
        False.

    """
    if isinstance(ind, (int, str)):
        allowed_ind_options = {"x", "X", 1, "y", "Y", 2, "z", "Z", 3, "i", "I", 0}
        if ind not in allowed_ind_options:
            raise ValueError(f"Invalid Pauli operator index provided. Allowed options are {allowed_ind_options}.")
        if ind in {"x", "X", 1}:
            pauli_mat = np.array([[0, 1], [1, 0]])
        elif ind in {"y", "Y", 2}:
            pauli_mat = np.array([[0, -1j], [1j, 0]])
        elif ind in {"z", "Z", 3}:
            pauli_mat = np.array([[1, 0], [0, -1]])
        else:
            pauli_mat = np.identity(2)

        if is_sparse:
            pauli_mat = csr_array(pauli_mat)

        return pauli_mat

    num_qubits = len(ind)
    pauli_mats = []
    for i in range(num_qubits):
        pauli_mats.append(pauli(ind[i], is_sparse))
    return tensor(pauli_mats)

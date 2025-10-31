"""Tests for the is_sic_povm helper."""

from importlib import import_module

import numpy as np
import pytest

from toqito.state_props import is_sic_povm

is_sic_module = import_module("toqito.state_props.is_sic_povm")


def tetrahedral_qubit_sic() -> list[np.ndarray]:
    """Return the standard qubit SIC vectors."""
    omega = np.exp(2j * np.pi / 3)
    return [
        np.array([0, 1], dtype=np.complex128),
        np.array([np.sqrt(2 / 3), 1 / np.sqrt(3)], dtype=np.complex128),
        np.array([np.sqrt(2 / 3), omega / np.sqrt(3)], dtype=np.complex128),
        np.array([np.sqrt(2 / 3), (omega ** 2) / np.sqrt(3)], dtype=np.complex128),
    ]


def qutrit_weyl_sic() -> list[np.ndarray]:
    """Generate a qutrit SIC via the Weyl–Heisenberg orbit of a fiducial."""
    fiducial = np.array([0, 1, -1], dtype=np.complex128)
    fiducial = fiducial / np.linalg.norm(fiducial)
    d = fiducial.size

    omega = np.exp(2j * np.pi / d)
    tau = -np.exp(np.pi * 1j / d)
    states = []
    for a in range(d):
        for b in range(d):
            shift = np.roll(np.eye(d, dtype=np.complex128), shift=b, axis=1)
            phase = np.diag(omega ** (a * np.arange(d)))
            displacement = (tau ** (a * b)) * phase @ shift
            states.append(displacement @ fiducial)
    return states


def ququart_weyl_sic() -> list[np.ndarray]:
    """Generate a ququart SIC via the Weyl–Heisenberg orbit."""
    fiducial = np.array(
        [
            0.1546957480124473 - 0.36979542704291186j,
            -0.17857741590112038 + 0.09266580852242757j,
            0.24759672956910453 + 0.7082536428225504j,
            0.22371505588036112 + 0.4311240198664474j,
        ],
        dtype=np.complex128,
    )
    fiducial = fiducial / np.linalg.norm(fiducial)
    d = fiducial.size

    omega = np.exp(2j * np.pi / d)
    tau = -np.exp(np.pi * 1j / d)
    states = []
    for a in range(d):
        for b in range(d):
            shift = np.roll(np.eye(d, dtype=np.complex128), shift=b, axis=1)
            phase = np.diag(omega ** (a * np.arange(d)))
            displacement = (tau ** (a * b)) * phase @ shift
            states.append(displacement @ fiducial)
    return states


def test_is_sic_povm_recognizes_qubit_tetrahedron():
    """The tetrahedral qubit configuration is a SIC."""
    assert is_sic_povm(tetrahedral_qubit_sic())


def test_is_sic_povm_recognizes_qutrit_sic():
    """A Weyl–Heisenberg qutrit SIC should be accepted."""
    assert is_sic_povm(qutrit_weyl_sic())


def test_is_sic_povm_recognizes_ququart_sic():
    """The dimension-4 SIC should be accepted."""
    assert is_sic_povm(ququart_weyl_sic())


def test_is_sic_povm_accepts_column_vectors():
    """Column-vector inputs are flattened during normalization."""
    column_vectors = [state.reshape(-1, 1) for state in tetrahedral_qubit_sic()]
    assert is_sic_povm(column_vectors)


def test_is_sic_povm_rejects_wrong_cardinality():
    """Fail when the number of vectors does not equal d^2."""
    states = tetrahedral_qubit_sic()[:-1]
    assert not is_sic_povm(states)


def test_is_sic_povm_raises_on_invalid_shape():
    """Matrix-shaped state inputs are rejected."""
    with pytest.raises(ValueError):
        is_sic_povm([np.eye(2, dtype=np.complex128)])


def test_is_sic_povm_rejects_non_constant_overlaps():
    """Small perturbations destroy the SIC property."""
    states = tetrahedral_qubit_sic()
    states[1] = states[1] + 0.01 * np.array([1, 0], dtype=np.complex128)
    assert not is_sic_povm(states)


def test_is_sic_povm_empty_input_raises():
    """At least one vector is required."""
    with pytest.raises(ValueError):
        is_sic_povm([])


def test_is_sic_povm_dimension_mismatch_raises():
    """Vectors of differing dimensions cause a ValueError."""
    e0 = np.array([1, 0], dtype=np.complex128)
    e2 = np.array([1, 0, 0], dtype=np.complex128)
    with pytest.raises(ValueError):
        is_sic_povm([e0, e2])


def test_is_sic_povm_zero_dimension_raises(monkeypatch):
    """A zero-dimensional normalized state triggers a ValueError."""
    monkeypatch.setattr(is_sic_module, "normalize", lambda *_args, **_kwargs: np.array([], dtype=np.complex128))
    with pytest.raises(ValueError):
        is_sic_povm([np.array([1], dtype=np.complex128)])


def test_is_sic_povm_detects_non_unit_gram(monkeypatch):
    """Fail when the Gram matrix has non-unit diagonal."""
    states = tetrahedral_qubit_sic()

    def fake_gram(_vectors):
        gram = np.eye(len(states), dtype=np.complex128)
        gram[1, 1] = 0.9
        return gram

    monkeypatch.setattr(is_sic_module, "vectors_to_gram_matrix", fake_gram)
    assert not is_sic_povm(states)


def test_is_sic_povm_raises_on_zero_vector():
    """Zero vectors are invalid inputs."""
    with pytest.raises(ValueError):
        is_sic_povm([np.zeros(2, dtype=np.complex128)])


def test_is_sic_povm_detects_frame_operator_failure(monkeypatch):
    """Catch failures in the frame-operator condition."""
    states = [np.array([1, 0], dtype=np.complex128) for _ in range(4)]

    def fake_gram(_vectors):
        val = np.sqrt(1 / 3)
        gram = np.full((4, 4), val, dtype=np.complex128)
        np.fill_diagonal(gram, 1.0)
        return gram

    monkeypatch.setattr(is_sic_module, "vectors_to_gram_matrix", fake_gram)
    assert not is_sic_povm(states)

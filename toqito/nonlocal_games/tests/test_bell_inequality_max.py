"""Tests for classical, quantum (NPA), and no-signalling maximums for Bell inequalities."""
import cvxpy
import numpy as np
import pytest

from toqito.helper.bell_npa_constraints import bell_npa_constraints
from toqito.nonlocal_games.bell_inequality_max import bell_inequality_max


@pytest.fixture(name="chsh_fc")
def chsh_fc_fixture():
    """Provide the CHSH inequality coefficients in FC notation."""
    return np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])


@pytest.fixture(name="chsh_cg")
def chsh_cg_fixture():
    """Provide the CHSH inequality coefficients in CG notation."""
    return np.array([[0, -1, 0], [-1, 1, 1], [0, 1, -1]])

@pytest.fixture(name="chsh_game_fp")
def chsh_game_fp_fixture():
    """Provide the coefficients for the CHSH *game* winning probability in FP notation."""
    chsh_fp = np.zeros((2, 2, 2, 2))
    chsh_fp[0, 0, 0, 0] = 0.25
    chsh_fp[1, 1, 0, 0] = 0.25
    chsh_fp[0, 0, 0, 1] = 0.25
    chsh_fp[1, 1, 0, 1] = 0.25
    chsh_fp[0, 0, 1, 0] = 0.25
    chsh_fp[1, 1, 1, 0] = 0.25
    chsh_fp[0, 1, 1, 1] = 0.25
    chsh_fp[1, 0, 1, 1] = 0.25
    return chsh_fp

@pytest.fixture(name="i3322_cg")
def i3322_cg_fixture():
    """Provide the coefficients for the I3322 Bell inequality in CG notation."""
    return np.array([[0, 1, 0, 0], [1, -1, -1, -1], [0, -1, -1, 1], [0, -1, 1, 0]])

@pytest.fixture(name="desc_chsh")
def desc_chsh_fixture():
    """Provide the scenario description list for the CHSH inequality/game."""
    return [2, 2, 2, 2]

@pytest.fixture(name="desc_i3322")
def desc_i3322_fixture():
    """Provide the scenario description list for the I3322 inequality."""
    return [2, 2, 3, 3]

RTOL = 1e-4
ATOL = 1e-4

def test_chsh_fc_classical(chsh_fc, desc_chsh):
    """Test classical maximum for CHSH inequality in FC notation."""
    assert bell_inequality_max(chsh_fc, desc_chsh, "fc", "classical") == pytest.approx(
        2.0, abs=ATOL
    )


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_chsh_fc_quantum(chsh_fc, desc_chsh):
    """Test quantum maximum (Tsirelson bound) for CHSH inequality in FC notation."""
    expected = 2 * np.sqrt(2)
    assert bell_inequality_max(chsh_fc, desc_chsh, "fc", "quantum", tol=1e-7) == pytest.approx(
        expected, rel=RTOL, abs=ATOL
    )


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_chsh_fc_nosignal(chsh_fc, desc_chsh):
    """Test no-signalling maximum for CHSH inequality in FC notation."""
    assert bell_inequality_max(chsh_fc, desc_chsh, "fc", "nosignal", tol=1e-9) == pytest.approx(
        4.0, abs=5e-5
    )


def test_chsh_cg_classical(chsh_cg, desc_chsh):
    """Test classical maximum for CHSH inequality in CG notation."""
    assert bell_inequality_max(chsh_cg, desc_chsh, "cg", "classical") == pytest.approx(
        0.0, abs=ATOL
    )


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_chsh_cg_quantum(chsh_cg, desc_chsh):
    """Test quantum maximum for CHSH inequality in CG notation."""
    expected = 1 / np.sqrt(2) - 0.5
    assert bell_inequality_max(chsh_cg, desc_chsh, "cg", "quantum", tol=1e-7) == pytest.approx(
        expected, rel=RTOL, abs=ATOL
    )


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_chsh_cg_nosignal(chsh_cg, desc_chsh):
    """Test no-signalling maximum for CHSH inequality in CG notation."""
    assert bell_inequality_max(chsh_cg, desc_chsh, "cg", "nosignal", tol=1e-9) == pytest.approx(
        0.5, abs=ATOL
    )


def test_chsh_game_fp_classical(chsh_game_fp, desc_chsh):
    """Test classical maximum for CHSH game winning probability in FP notation."""
    assert bell_inequality_max(
        chsh_game_fp, desc_chsh, "fp", "classical"
    ) == pytest.approx(0.75, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_chsh_game_fp_quantum(chsh_game_fp, desc_chsh):
    """Test quantum maximum for CHSH game winning probability in FP notation."""
    expected = (1 + 1 / np.sqrt(2)) / 2
    assert bell_inequality_max(
        chsh_game_fp, desc_chsh, "fp", "quantum", k="1+ab", tol=1e-7
    ) == pytest.approx(expected, rel=RTOL, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_chsh_game_fp_nosignal(chsh_game_fp, desc_chsh):
    """Test no-signalling maximum for CHSH game winning probability in FP notation."""
    assert bell_inequality_max(
        chsh_game_fp, desc_chsh, "fp", "nosignal", tol=1e-9
    ) == pytest.approx(1.0, abs=ATOL)


def test_i3322_cg_classical(i3322_cg, desc_i3322):
    """Test classical maximum for I3322 inequality in CG notation."""
    assert bell_inequality_max(
        i3322_cg, desc_i3322, "cg", "classical"
    ) == pytest.approx(1.0, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
@pytest.mark.parametrize(
    "k_level, expected_val_toqito",
    [
        (1, 1.375),
        ("1+ab", 1.26870172),
        (2, 1.25538020),
    ],
)
def test_i3322_cg_quantum(i3322_cg, desc_i3322, k_level, expected_val_toqito):
    """Test quantum maximum for I3322 inequality in CG notation using various NPA levels."""
    sdp_tol = 1e-7
    assert bell_inequality_max(
        i3322_cg, desc_i3322, "cg", "quantum", k=k_level, tol=sdp_tol
    ) == pytest.approx(expected_val_toqito, rel=RTOL, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_i3322_cg_nosignal(i3322_cg, desc_i3322):
    """Test no-signalling maximum for I3322 inequality in CG notation."""
    assert bell_inequality_max(
        i3322_cg, desc_i3322, "cg", "nosignal", tol=1e-9
    ) == pytest.approx(2.0, abs=ATOL)


def test_classical_swap_fc():
    """Test classical max is invariant under swapping Alice/Bob roles (FC notation)."""
    desc_32 = [2, 2, 3, 2]
    dummy_fc_32 = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1]
    ])
    val_no_swap = bell_inequality_max(dummy_fc_32, desc_32, "fc", "classical")

    desc_23 = [2, 2, 2, 3]
    dummy_fc_23 = dummy_fc_32.T
    val_swap = bell_inequality_max(dummy_fc_23, desc_23, "fc", "classical")

    assert val_swap == pytest.approx(val_no_swap, abs=ATOL)


def test_classical_swap_fp():
    """Test classical max is invariant under swapping Alice/Bob roles (FP notation, non-binary)."""
    desc_swap = [2, 3, 2, 1]
    dummy_fp_swap = np.random.rand(2, 3, 2, 1)
    sum_ab = np.sum(dummy_fp_swap, axis=(0, 1), keepdims=True)
    sum_ab[sum_ab == 0] = 1
    dummy_fp_swap /= sum_ab

    val_swap = bell_inequality_max(dummy_fp_swap, desc_swap, "fp", "classical")

    desc_noswap = [3, 2, 1, 2]
    dummy_fp_noswap = np.transpose(dummy_fp_swap, (1, 0, 3, 2))
    val_noswap = bell_inequality_max(dummy_fp_noswap, desc_noswap, "fp", "classical")

    assert val_swap == pytest.approx(val_noswap, abs=ATOL)


def test_invalid_notation(chsh_fc, desc_chsh):
    """Test ValueError is raised for invalid notation string."""
    with pytest.raises(ValueError, match="Invalid notation"):
        bell_inequality_max(chsh_fc, desc_chsh, "invalid", "classical")


def test_invalid_mtype(chsh_fc, desc_chsh):
    """Test ValueError is raised for invalid mtype string."""
    with pytest.raises(ValueError, match="Invalid mtype"):
        bell_inequality_max(chsh_fc, desc_chsh, "fc", "invalid_type")


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_infeasible_setup_via_call():
    """Test solver detects infeasibility when manually added constraints make it so."""
    desc = [2, 2, 1, 1]
    coeffs_cg = np.array([[0, 0], [0, 1]])

    cg_dim = ((desc[0] - 1) * desc[2] + 1, (desc[1] - 1) * desc[3] + 1)
    p_var = cvxpy.Variable(cg_dim, name="p_infeas_test")
    objective = cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(coeffs_cg, p_var)))

    constraints = [p_var[0, 0] == 1, p_var[1, 1] >= 1.1]
    try:
        constraints += bell_npa_constraints(p_var, desc, k=1)
    except ValueError:
        pytest.fail("NPA constraint generation failed unexpectedly.")

    problem = cvxpy.Problem(objective, constraints)
    problem.solve(solver=cvxpy.SCS, eps=1e-8)

    assert problem.status in [cvxpy.INFEASIBLE, cvxpy.INFEASIBLE_INACCURATE]


def test_classical_nonbinary_fc_input_error():
    """Test ValueError for classical calculation with FC notation and non-binary outputs."""
    desc = [3, 2, 2, 2]
    dummy_fc = np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    with pytest.raises(ValueError, match="Notation conversion failed for non-binary scenario"):
        bell_inequality_max(dummy_fc, desc, "fc", "classical")


def test_classical_no_bob_inputs():
    """Test classical calculation when Bob has zero inputs (mb=0)."""
    desc_2x2_mb0 = [2, 2, 2, 0]
    coeffs_fc = np.array([[0], [1], [-1]])
    val_fc = bell_inequality_max(coeffs_fc, desc_2x2_mb0, "fc", "classical")
    expected_fc = coeffs_fc[0, 0] + np.sum(np.abs(coeffs_fc[1:, 0]))
    assert val_fc == pytest.approx(expected_fc, abs=ATOL)

    desc_gen_mb0 = [3, 2, 2, 0]
    coeffs_fp_gen = np.zeros((3, 2, 2, 0))
    val_fp = bell_inequality_max(coeffs_fp_gen, desc_gen_mb0, "fp", "classical")
    assert val_fp == pytest.approx(0.0, abs=ATOL)


def test_classical_no_alice_inputs():
    """Test classical calculation when Alice has zero inputs (ma=0)."""
    desc_2x2_ma0 = [2, 2, 0, 2]
    coeffs_fc = np.array([[0, 1, -1]])
    val_fc = bell_inequality_max(coeffs_fc, desc_2x2_ma0, "fc", "classical")
    expected_fc = coeffs_fc[0, 0] + np.sum(np.abs(coeffs_fc[0, 1:]))
    assert val_fc == pytest.approx(expected_fc, abs=ATOL)

    desc_gen_ma0 = [2, 3, 0, 2]
    coeffs_fp_gen = np.zeros((2, 3, 0, 2))
    val_fp = bell_inequality_max(coeffs_fp_gen, desc_gen_ma0, "fp", "classical")
    assert val_fp == pytest.approx(0.0, abs=ATOL)


def test_classical_binary_ma0_fp():
    """Test classical calculation with ma=0 using FP notation for a specific case."""
    desc_ma0 = [2, 2, 0, 2]
    coeffs_fp = np.zeros((2, 2, 0, 2))
    coeffs_fp[0, 0, :, 0] = 0.5 # p(00|_, 0) = 0.5
    coeffs_fp[1, 1, :, 0] = 0.5 # p(11|_, 0) = 0.5
    coeffs_fp[0, 1, :, 1] = 0.5 # p(01|_, 1) = 0.5
    coeffs_fp[1, 0, :, 1] = 0.5 # p(10|_, 1) = 0.5
    assert bell_inequality_max(coeffs_fp, desc_ma0, "fp", "classical") == pytest.approx(0.0, abs=ATOL)


def test_classical_binary_mb0_cg():
    """Test classical calculation with mb=0 using CG notation."""
    desc_mb0 = [2, 2, 2, 0]
    coeffs_cg = np.array([[0], [-1], [1]]) # pA(0|0)=-1, pA(0|1)=1
    assert bell_inequality_max(coeffs_cg, desc_mb0, "cg", "classical") == pytest.approx(1.0, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_oa1():
    """Test no-signalling calculation when Alice has only one output (oa=1)."""
    desc = [1, 2, 2, 2]
    coeffs_cg = np.array([[1, 0.5, 0.5]])
    assert bell_inequality_max(coeffs_cg, desc, "cg", "nosignal", tol=1e-9) == pytest.approx(2.0, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_ob1():
    """Test no-signalling calculation when Bob has only one output (ob=1)."""
    desc = [2, 1, 2, 2]
    coeffs_cg = np.array([[1], [0.5], [0.5]])
    assert bell_inequality_max(coeffs_cg, desc, "cg", "nosignal", tol=1e-9) == pytest.approx(2.0, abs=ATOL)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_fc_nonbinary_error():
    """Test ValueError for no-signalling calculation with FC notation and non-binary outputs."""
    desc = [3, 2, 2, 2]
    dummy_fc = np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    with pytest.raises(ValueError, match="Notation conversion failed: 'fc' notation is only supported"):
        bell_inequality_max(dummy_fc, desc, "fc", "nosignal")


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_fc_nonbinary_error():
    """Test ValueError for quantum calculation with FC notation and non-binary outputs."""
    desc = [3, 2, 2, 2]
    dummy_fc = np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    with pytest.raises(ValueError, match="Notation conversion failed: 'fc' notation is only supported"):
        bell_inequality_max(dummy_fc, desc, "fc", "quantum")


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_invalid_k_error(chsh_cg, desc_chsh):
    """Test ValueError is raised for invalid NPA level 'k' inputs."""
    with pytest.raises(ValueError, match="Invalid NPA level k=-1"):
        bell_inequality_max(chsh_cg, desc_chsh, "cg", "quantum", k=-1)
    with pytest.raises(ValueError, match=r"Invalid NPA level k='invalid_level'"):
        bell_inequality_max(chsh_cg, desc_chsh, "cg", "quantum", k="invalid_level")
    with pytest.raises(ValueError, match=r"Invalid NPA level k='1\+abc'"):
        bell_inequality_max(chsh_cg, desc_chsh, "cg", "quantum", k="1+abc")


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_infeasible_status(mocker, capfd):
    """Test handling of infeasible solver status for no-signalling."""
    mock_problem = mocker.Mock(spec=cvxpy.Problem)
    mock_problem.solve.return_value = -np.inf
    mock_problem.status = cvxpy.INFEASIBLE
    mocker.patch("cvxpy.Problem", return_value=mock_problem)

    desc = [2, 2, 1, 1]
    coeffs_cg = np.array([[0, 0], [0, 1]])
    result = bell_inequality_max(coeffs_cg, desc, "cg", "nosignal")
    captured = capfd.readouterr()
    assert result == -np.inf
    assert "Warning: Solver status for 'nosignal': infeasible" in captured.out


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_unbounded_status(mocker, capfd):
    """Test handling of unbounded solver status for no-signalling."""
    mock_problem = mocker.Mock(spec=cvxpy.Problem)
    mock_problem.solve.return_value = np.inf
    mock_problem.status = cvxpy.UNBOUNDED
    mocker.patch("cvxpy.Problem", return_value=mock_problem)

    desc = [2, 2, 1, 1]
    coeffs_cg = np.array([[0, 0], [0, 1]])
    result = bell_inequality_max(coeffs_cg, desc, "cg", "nosignal")
    captured = capfd.readouterr()
    assert result == np.inf
    assert "Warning: Solver status for 'nosignal': unbounded" in captured.out


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_infeasible_status(mocker, capfd):
    """Test handling of infeasible solver status for quantum."""
    mock_problem = mocker.Mock(spec=cvxpy.Problem)
    mock_problem.solve.return_value = -np.inf
    mock_problem.status = cvxpy.INFEASIBLE_INACCURATE
    mocker.patch("cvxpy.Problem", return_value=mock_problem)

    desc = [2, 2, 2, 2]
    coeffs_cg = np.array([[0, -1, 0], [-1, 1, 1], [0, 1, -1]])
    result = bell_inequality_max(coeffs_cg, desc, "cg", "quantum")
    captured = capfd.readouterr()
    assert result == -np.inf
    assert "Warning: Solver status for 'quantum' k=1: infeasible_inaccurate" in captured.out


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_unbounded_status(mocker, capfd):
    """Test handling of unbounded solver status for quantum."""
    mock_problem = mocker.Mock(spec=cvxpy.Problem)
    mock_problem.solve.return_value = np.inf
    mock_problem.status = cvxpy.UNBOUNDED_INACCURATE
    mocker.patch("cvxpy.Problem", return_value=mock_problem)

    desc = [2, 2, 2, 2]
    coeffs_cg = np.array([[0, -1, 0], [-1, 1, 1], [0, 1, -1]])
    result = bell_inequality_max(coeffs_cg, desc, "cg", "quantum")
    captured = capfd.readouterr()
    assert result == np.inf
    assert "Warning: Solver status for 'quantum' k=1: unbounded_inaccurate" in captured.out


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_npa_error(mocker, chsh_cg, desc_chsh):
    """Test ValueError is raised if bell_npa_constraints fails."""
    mocker.patch(
        "toqito.nonlocal_games.bell_inequality_max.bell_npa_constraints",
        side_effect=ValueError("Mock NPA Error"),
    )
    with pytest.raises(ValueError, match="Error generating NPA constraints: Mock NPA Error"):
        bell_inequality_max(chsh_cg, desc_chsh, "cg", "quantum", k=1)


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_result_is_nan(mocker, chsh_cg, desc_chsh):
    """Test return value is -inf if solver returns NaN for quantum."""
    mock_problem = mocker.Mock(spec=cvxpy.Problem)
    mock_problem.solve.return_value = np.nan
    mock_problem.status = cvxpy.SOLVER_ERROR
    mocker.patch("cvxpy.Problem", return_value=mock_problem)
    assert bell_inequality_max(chsh_cg, desc_chsh, "cg", "quantum") == -np.inf


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_result_is_nan(mocker, chsh_cg, desc_chsh):
    """Test return value is -inf if solver returns None/NaN for no-signalling."""
    mock_problem = mocker.Mock(spec=cvxpy.Problem)
    mock_problem.solve.return_value = None
    mock_problem.status = cvxpy.SOLVER_ERROR
    mocker.patch("cvxpy.Problem", return_value=mock_problem)
    assert bell_inequality_max(chsh_cg, desc_chsh, "cg", "nosignal") == -np.inf


def test_classical_cg_conversion_error(desc_chsh):
    """Test ValueError for classical calculation if CG coefficients have wrong shape (binary)."""
    invalid_coeffs_cg = np.zeros((3, 4)) # Correct shape is (3, 3)
    with pytest.raises(ValueError, match="Notation conversion failed for binary scenario: CG coefficient shape"):
        bell_inequality_max(invalid_coeffs_cg, desc_chsh, "cg", "classical")


def test_classical_nonbinary_cg_conversion_error():
    """Test ValueError for classical calculation if CG coefficients have wrong shape (non-binary)."""
    desc_nonbin = [3, 2, 2, 2]
    invalid_coeffs_cg = np.zeros((5, 4)) # Correct shape is (5, 3)
    with pytest.raises(ValueError, match="Notation conversion failed for non-binary scenario: CG coefficient shape"):
        bell_inequality_max(invalid_coeffs_cg, desc_nonbin, "cg", "classical")


def test_classical_binary_fp_conversion_error(desc_chsh):
    """Test ValueError for classical calculation if FP coefficients have wrong shape (binary)."""
    invalid_coeffs_fp = np.zeros((2, 2, 3, 2)) # Correct shape is (2, 2, 2, 2)
    with pytest.raises(ValueError, match="Notation conversion failed for binary scenario: FP coefficient shape"):
        bell_inequality_max(invalid_coeffs_fp, desc_chsh, "fp", "classical")


def test_classical_nonbinary_ob1():
    """Test classical calculation with non-binary Alice and ob=1 using FP."""
    desc = [2, 1, 2, 2]
    coeffs_fp = np.zeros((2, 1, 2, 2))
    coeffs_fp[0, 0, 0, 0] = 1
    coeffs_fp[0, 0, 1, 1] = 1
    assert bell_inequality_max(coeffs_fp, desc, "fp", "classical") == pytest.approx(2.0, abs=ATOL)

    coeffs_fp_2 = np.zeros((2, 1, 2, 2))
    coeffs_fp_2[0, 0, 0, 0] = 0.8
    coeffs_fp_2[1, 0, 0, 0] = 0.2
    coeffs_fp_2[0, 0, 1, 1] = 1.0
    assert bell_inequality_max(coeffs_fp_2, desc, "fp", "classical") == pytest.approx(1.8, abs=ATOL)


def test_classical_empty_fp():
    """Test classical calculation returns 0.0 if FP tensor is empty (ma=0 or mb=0)."""
    desc_ma0 = [2, 3, 0, 2]
    coeffs_fp_ma0 = np.zeros((2, 3, 0, 2))
    assert bell_inequality_max(coeffs_fp_ma0, desc_ma0, "fp", "classical") == pytest.approx(0.0, abs=ATOL)

    desc_mb0 = [3, 2, 2, 0]
    coeffs_fp_mb0 = np.zeros((3, 2, 2, 0))
    assert bell_inequality_max(coeffs_fp_mb0, desc_mb0, "fp", "classical") == pytest.approx(0.0, abs=ATOL)


def test_classical_nonbinary_fp_dense():
    """Test classical calculation runs with dense, random, non-binary FP coefficients."""
    desc_nonbin = [3, 2, 2, 2]
    coeffs_fp = np.random.rand(3, 2, 2, 2)
    sum_ab = np.sum(coeffs_fp, axis=(0, 1), keepdims=True)
    sum_ab[sum_ab == 0] = 1
    coeffs_fp /= sum_ab
    result = bell_inequality_max(coeffs_fp, desc_nonbin, "fp", "classical")
    assert isinstance(result, float)
    assert result >= 0


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_nosignal_cg_shape_error(desc_chsh):
    """Test ValueError for no-signalling calculation if CG coefficients have wrong shape."""
    invalid_coeffs_cg = np.zeros((3, 4)) # Correct shape is (3, 3)
    with pytest.raises(ValueError, match="Coefficient shape"):
        bell_inequality_max(invalid_coeffs_cg, desc_chsh, "cg", "nosignal")


@pytest.mark.skipif(
    cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed."
)
def test_quantum_cg_shape_error(desc_chsh):
    """Test ValueError for quantum calculation if CG coefficients have wrong shape."""
    invalid_coeffs_cg = np.zeros((3, 4)) # Correct shape is (3, 3)
    with pytest.raises(ValueError, match="Coefficient shape"):
        bell_inequality_max(invalid_coeffs_cg, desc_chsh, "cg", "quantum")


def test_classical_fc_shape_error(desc_chsh):
    """Test ValueError for classical calculation if FC coefficients have wrong shape."""
    invalid_coeffs_fc = np.zeros((3, 4)) # Correct shape is (3, 3)
    with pytest.raises(ValueError, match="Notation conversion failed for binary scenario: FC coefficient shape"):
        bell_inequality_max(invalid_coeffs_fc, desc_chsh, "fc", "classical")


def test_classical_nonbinary_fp_shape_error():
    """Test ValueError for classical calculation if non-binary FP coefficients have wrong shape."""
    desc_nonbin = [3, 2, 2, 2]
    invalid_coeffs_fp = np.zeros((3, 2, 3, 2)) # Correct shape is (3, 2, 2, 2)
    with pytest.raises(ValueError, match="Notation conversion failed for non-binary scenario: FP coefficient shape"):
        bell_inequality_max(invalid_coeffs_fp, desc_nonbin, "fp", "classical")


def test_classical_nonbinary_swap_triggered():
    """Test classical non-binary calculation executes when party swap is triggered."""
    desc = [2, 3, 1, 2]
    coeffs_fp = np.random.rand(2, 3, 1, 2)
    sum_ab = np.sum(coeffs_fp, axis=(0, 1), keepdims=True)
    sum_ab[sum_ab == 0] = 1
    coeffs_fp /= sum_ab
    result = bell_inequality_max(coeffs_fp, desc, "fp", "classical")
    assert isinstance(result, float)


@pytest.mark.skipif(cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed.")
def test_nosignal_fp_conversion_internal_error(mocker, desc_chsh):
    """Test catching internal ValueError from fp_to_cg in nosignal path."""
    dummy_fp_coeffs = np.zeros((2, 2, 2, 2)) # Correct shape
    mocker.patch(
        "toqito.nonlocal_games.bell_inequality_max.fp_to_cg",
        side_effect=ValueError("Internal fp_to_cg error")
    )

    with pytest.raises(ValueError, match="Notation conversion failed: Internal fp_to_cg error"):
        bell_inequality_max(dummy_fp_coeffs, desc_chsh, "fp", "nosignal")

@pytest.mark.skipif(cvxpy.SCS not in cvxpy.installed_solvers(), reason="SCS solver not installed.")
def test_quantum_fc_conversion_internal_error(mocker, chsh_fc, desc_chsh):
    """Test catching internal ValueError from fc_to_cg in quantum path."""
    mocker.patch(
        "toqito.nonlocal_games.bell_inequality_max.fc_to_cg",
        side_effect=ValueError("Internal fc_to_cg error")
    )
    with pytest.raises(ValueError, match="Notation conversion failed: Internal fc_to_cg error"):
        bell_inequality_max(chsh_fc, desc_chsh, "fc", "quantum")

def test_classical_nonbinary_cg_to_fp_internal_error(mocker):
    """Test catching internal ValueError from cg_to_fp in classical non-binary path."""
    desc_actual_nonbin = [3, 3, 2, 2]
    coeffs_cg_actual_nonbin = np.zeros(((3 - 1) * 2 + 1, (3 - 1) * 2 + 1))
    mocker.patch(
        "toqito.nonlocal_games.bell_inequality_max.cg_to_fp",
        side_effect=ValueError("Internal cg_to_fp error")
    )

    with pytest.raises(ValueError, match="Notation conversion failed for non-binary scenario: Internal cg_to_fp error"):
        bell_inequality_max(coeffs_cg_actual_nonbin, desc_actual_nonbin, "cg", "classical")

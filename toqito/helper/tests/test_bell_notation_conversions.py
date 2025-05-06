"""Test bell_notation_conversions."""

import numpy as np
import pytest

from toqito.helper.bell_notation_conversions import cg_to_fc, cg_to_fp, fc_to_cg, fc_to_fp, fp_to_cg, fp_to_fc


@pytest.fixture(name="chsh_cg")
def fixture_chsh_cg():
    """CHSH functional in CG notation."""
    return np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])


@pytest.fixture(name="chsh_fc")
def fixture_chsh_fc():
    """CHSH functional in FC notation."""
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.25, -0.25], [0.0, -0.25, 0.25]])


@pytest.fixture(name="chsh_fp_func")
def fixture_chsh_fp_func(chsh_fc):
    """CHSH functional in FP notation (derived from FC)."""
    return fc_to_fp(chsh_fc, behavior=False)


@pytest.fixture(name="chsh_fp_func_doc")
def fixture_chsh_fp_func_doc():
    """CHSH functional in FP notation (from fp_to_cg docstring and derived from chsh_cg)."""
    fp_mat = np.zeros((2, 2, 2, 2))
    fp_mat[0, 0, 0, 0] = 1
    fp_mat[0, 0, 0, 1] = -1
    fp_mat[0, 0, 1, 0] = -1
    fp_mat[0, 0, 1, 1] = 1
    return fp_mat


@pytest.fixture(name="uniform_cg_bhv")
def fixture_uniform_cg_bhv():
    """Uniform probability distribution (behavior) in CG notation."""
    return np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])


@pytest.fixture(name="uniform_fc_bhv")
def fixture_uniform_fc_bhv():
    """Uniform probability distribution (behavior) in FC notation."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.fixture(name="uniform_fp_bhv")
def fixture_uniform_fp_bhv():
    """Uniform probability distribution (behavior) in FP notation."""
    return np.full((2, 2, 2, 2), 0.25)


@pytest.fixture(name="pr_box_fp")
def fixture_pr_box_fp():
    """Define the PR box in FP notation."""
    pr_box = np.zeros((2, 2, 2, 2))
    pr_box[0, 0, 0, 0] = 0.5
    pr_box[1, 1, 0, 0] = 0.5
    pr_box[0, 0, 0, 1] = 0.5
    pr_box[1, 1, 0, 1] = 0.5
    pr_box[0, 0, 1, 0] = 0.5
    pr_box[1, 1, 1, 0] = 0.5
    pr_box[0, 1, 1, 1] = 0.5
    pr_box[1, 0, 1, 1] = 0.5
    return pr_box


@pytest.fixture(name="pr_box_cg_qetlab")
def fixture_pr_box_cg_qetlab():
    """PR box in CG notation using QETLAB convention (expected result from fp_to_cg)."""
    return np.array([[1.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]])


@pytest.fixture(name="pr_box_fc_qetlab")
def fixture_pr_box_fc_qetlab():
    """PR box in FC notation using QETLAB convention (expected result from fp_to_fc)."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]])


@pytest.fixture(name="desc_2222")
def fixture_desc_2222():
    """Return standard descriptor [oa, ob, ia, ib]."""
    return [2, 2, 2, 2]


@pytest.fixture(name="desc_3221")
def fixture_desc_3221():
    """Non-binary descriptor [oa, ob, ia, ib]."""
    return [3, 2, 2, 1]


@pytest.fixture(name="cg_3221_bhv")
def fixture_cg_3221_bhv():
    """Non-binary behavior in CG notation."""
    return np.array([[1.0, 0.6], [0.4, 0.2], [0.3, 0.1], [0.5, 0.3], [0.2, 0.05]])


@pytest.fixture(name="fp_3221_bhv")
def fixture_fp_3221_bhv(cg_3221_bhv, desc_3221):
    """Non-binary behavior in FP notation (derived from CG)."""
    return cg_to_fp(cg_3221_bhv, desc_3221, behavior=True)


def test_cg_to_fc_functional(chsh_cg, chsh_fc):
    """Test cg_to_fc for functional case."""
    np.testing.assert_allclose(cg_to_fc(chsh_cg, behavior=False), chsh_fc)


def test_cg_to_fc_behavior(uniform_cg_bhv, uniform_fc_bhv):
    """Test cg_to_fc for behavior case."""
    np.testing.assert_allclose(cg_to_fc(uniform_cg_bhv, behavior=True), uniform_fc_bhv)


def test_fc_to_cg_functional(chsh_fc, chsh_cg):
    """Test fc_to_cg for functional case."""
    np.testing.assert_allclose(fc_to_cg(chsh_fc, behavior=False), chsh_cg)


def test_fc_to_cg_behavior(uniform_fc_bhv, uniform_cg_bhv):
    """Test fc_to_cg for behavior case."""
    np.testing.assert_allclose(fc_to_cg(uniform_fc_bhv, behavior=True), uniform_cg_bhv)


def test_cg_to_fp_functional(chsh_cg, desc_2222, chsh_fp_func_doc):
    """Test cg_to_fp for functional case."""
    np.testing.assert_allclose(cg_to_fp(chsh_cg, desc_2222, behavior=False), chsh_fp_func_doc)


def test_cg_to_fp_behavior(uniform_cg_bhv, desc_2222, uniform_fp_bhv):
    """Test cg_to_fp for behavior case."""
    np.testing.assert_allclose(cg_to_fp(uniform_cg_bhv, desc_2222, behavior=True), uniform_fp_bhv)


def test_cg_to_fp_behavior_non_binary(cg_3221_bhv, desc_3221, fp_3221_bhv):
    """Test cg_to_fp for behavior case with non-binary outcomes/inputs."""
    np.testing.assert_allclose(cg_to_fp(cg_3221_bhv, desc_3221, behavior=True), fp_3221_bhv)


def test_cg_to_fp_functional_zero_inputs():
    """Test cg_to_fp functional with zero inputs (edge case for coverage)."""
    cg_mat = np.array([[5.0]])
    desc_ia0 = [2, 2, 0, 2]
    desc_ib0 = [2, 2, 2, 0]

    expected_fp_ia0 = np.zeros((2, 2, 0, 2))
    expected_fp_ib0 = np.zeros((2, 2, 2, 0))
    np.testing.assert_allclose(cg_to_fp(cg_mat, desc_ia0, behavior=False), expected_fp_ia0)
    np.testing.assert_allclose(cg_to_fp(cg_mat, desc_ib0, behavior=False), expected_fp_ib0)


def test_cg_to_fp_behavior_one_output_alice():
    """Test cg_to_fp behavior with oa=1 (edge case for coverage)."""
    desc = [1, 2, 1, 1]
    cg_mat = np.array([[1.0, 0.6]])

    expected_fp = np.zeros((1, 2, 1, 1))
    expected_fp[0, 0, 0, 0] = cg_mat[0, 1]
    expected_fp[0, 1, 0, 0] = cg_mat[0, 0] - cg_mat[0, 1]
    np.testing.assert_allclose(cg_to_fp(cg_mat, desc, behavior=True), expected_fp)


def test_cg_to_fp_behavior_one_output_bob():
    """Test cg_to_fp behavior with ob=1 (edge case for coverage)."""
    desc = [2, 1, 1, 1]
    cg_mat = np.array([[1.0], [0.4]])

    expected_fp = np.zeros((2, 1, 1, 1))
    expected_fp[0, 0, 0, 0] = cg_mat[1, 0]
    expected_fp[1, 0, 0, 0] = cg_mat[0, 0] - cg_mat[1, 0]
    np.testing.assert_allclose(cg_to_fp(cg_mat, desc, behavior=True), expected_fp)


def test_fc_to_fp_functional(chsh_fc, chsh_fp_func):
    """Test fc_to_fp for functional case."""
    np.testing.assert_allclose(fc_to_fp(chsh_fc, behavior=False), chsh_fp_func)


def test_fc_to_fp_behavior(uniform_fc_bhv, uniform_fp_bhv):
    """Test fc_to_fp for behavior case."""
    np.testing.assert_allclose(fc_to_fp(uniform_fc_bhv, behavior=True), uniform_fp_bhv)


def test_fc_to_fp_behavior_pr_box(pr_box_fc_qetlab, pr_box_fp):
    """Test fc_to_fp for PR box behavior using QETLAB FC convention."""
    calculated_fp = fc_to_fp(pr_box_fc_qetlab, behavior=True)
    np.testing.assert_allclose(calculated_fp, pr_box_fp, atol=1e-8)


def test_fc_to_fp_functional_zero_inputs():
    """Test fc_to_fp functional with zero inputs (edge case for coverage)."""
    fc_mat_ia0 = np.array([[5.0, 1.0, 2.0]])
    fc_mat_ib0 = np.array([[5.0], [1.0], [2.0]])

    expected_fp_ia0 = np.zeros((2, 2, 0, 2))
    expected_fp_ib0 = np.zeros((2, 2, 2, 0))
    np.testing.assert_allclose(fc_to_fp(fc_mat_ia0, behavior=False), expected_fp_ia0)
    np.testing.assert_allclose(fc_to_fp(fc_mat_ib0, behavior=False), expected_fp_ib0)


def test_fp_to_cg_functional(chsh_fp_func_doc, chsh_cg):
    """Test fp_to_cg for functional case using docstring FP."""
    np.testing.assert_allclose(fp_to_cg(chsh_fp_func_doc, behavior=False), chsh_cg)


def test_fp_to_cg_behavior_uniform(uniform_fp_bhv, uniform_cg_bhv):
    """Test fp_to_cg for uniform behavior case."""
    np.testing.assert_allclose(fp_to_cg(uniform_fp_bhv, behavior=True), uniform_cg_bhv, atol=1e-8)


def test_fp_to_cg_behavior_pr_box(pr_box_fp, pr_box_cg_qetlab):
    """Test fp_to_cg for behavior case (PR box) using QETLAB convention."""
    np.testing.assert_allclose(fp_to_cg(pr_box_fp, behavior=True), pr_box_cg_qetlab, atol=1e-8)


def test_fp_to_cg_behavior_non_binary(fp_3221_bhv, cg_3221_bhv):
    """Test fp_to_cg for non-binary behavior case using QETLAB convention."""
    np.testing.assert_allclose(fp_to_cg(fp_3221_bhv, behavior=True), cg_3221_bhv, atol=1e-8)


def test_fp_to_cg_functional_one_output_alice():
    """Test fp_to_cg functional with oa=1 (edge case for coverage)."""
    v_mat = np.zeros((1, 2, 1, 1))
    v_mat[0, 0, 0, 0] = 0.6
    v_mat[0, 1, 0, 0] = 0.4

    expected_cg = np.zeros((1, 2))
    expected_cg[0, 0] = v_mat[0, 1, 0, 0]
    expected_cg[0, 1] = v_mat[0, 0, 0, 0] - v_mat[0, 1, 0, 0]
    np.testing.assert_allclose(fp_to_cg(v_mat, behavior=False), expected_cg)


def test_fp_to_cg_functional_one_output_bob():
    """Test fp_to_cg functional with ob=1 (edge case for coverage)."""
    v_mat = np.zeros((2, 1, 1, 1))
    v_mat[0, 0, 0, 0] = 0.4
    v_mat[1, 0, 0, 0] = 0.6
    expected_cg = np.zeros((2, 1))

    expected_cg[0, 0] = v_mat[1, 0, 0, 0]
    expected_cg[1, 0] = v_mat[0, 0, 0, 0] - v_mat[1, 0, 0, 0]
    np.testing.assert_allclose(fp_to_cg(v_mat, behavior=False), expected_cg)


def test_fp_to_cg_behavior_one_output_alice():
    """Test fp_to_cg behavior with oa=1 (edge case for coverage)."""
    v_mat = np.zeros((1, 2, 1, 1))
    v_mat[0, 0, 0, 0] = 0.6
    v_mat[0, 1, 0, 0] = 0.4

    expected_cg = np.zeros((1, 2))
    expected_cg[0, 0] = 1.0
    expected_cg[0, 1] = v_mat[0, 0, 0, 0]
    np.testing.assert_allclose(fp_to_cg(v_mat, behavior=True), expected_cg)


def test_fp_to_cg_behavior_one_output_bob():
    """Test fp_to_cg behavior with ob=1 (edge case for coverage)."""
    v_mat = np.zeros((2, 1, 1, 1))
    v_mat[0, 0, 0, 0] = 0.4
    v_mat[1, 0, 0, 0] = 0.6

    expected_cg = np.zeros((2, 1))
    expected_cg[0, 0] = 1.0
    expected_cg[1, 0] = v_mat[0, 0, 0, 0]
    np.testing.assert_allclose(fp_to_cg(v_mat, behavior=True), expected_cg)


def test_fp_to_cg_behavior_zero_inputs():
    """Test fp_to_cg behavior with zero inputs (edge case for coverage)."""
    v_mat_ia0 = np.zeros((2, 2, 0, 1))
    v_mat_ib0 = np.zeros((2, 2, 1, 0))

    expected_cg_ia0 = np.array([[1.0, 0.0]])
    expected_cg_ib0 = np.array([[1.0], [0.0]])
    np.testing.assert_allclose(fp_to_cg(v_mat_ia0, behavior=True), expected_cg_ia0)
    np.testing.assert_allclose(fp_to_cg(v_mat_ib0, behavior=True), expected_cg_ib0)


def test_fp_to_cg_functional_zero_outputs():
    """Test fp_to_cg functional with zero outputs (edge case for coverage)."""
    v_mat_oa0 = np.zeros((0, 2, 1, 1))

    expected_cg_oa0 = np.zeros((0, 2))
    np.testing.assert_allclose(fp_to_cg(v_mat_oa0, behavior=False), expected_cg_oa0)

    v_mat_ob0 = np.zeros((2, 0, 1, 1))

    expected_cg_ob0 = np.zeros((2, 0))
    np.testing.assert_allclose(fp_to_cg(v_mat_ob0, behavior=False), expected_cg_ob0)

    v_mat_oa0_ob0 = np.zeros((0, 0, 1, 1))

    expected_cg_oa0_ob0 = np.zeros((0, 0))
    np.testing.assert_allclose(fp_to_cg(v_mat_oa0_ob0, behavior=False), expected_cg_oa0_ob0)


def test_fp_to_cg_behavior_zero_outputs_exception():
    """Test fp_to_cg behavior raises error for zero outputs."""
    v_mat_oa0 = np.zeros((0, 2, 1, 1))
    v_mat_ob0 = np.zeros((2, 0, 1, 1))
    with pytest.raises(ValueError, match="behavior case requires non-zero outputs"):
        fp_to_cg(v_mat_oa0, behavior=True)
    with pytest.raises(ValueError, match="behavior case requires non-zero outputs"):
        fp_to_cg(v_mat_ob0, behavior=True)


def test_fp_to_fc_functional(chsh_fp_func, chsh_fc):
    """Test fp_to_fc for functional case."""
    np.testing.assert_allclose(fp_to_fc(chsh_fp_func, behavior=False), chsh_fc, atol=1e-8)


def test_fp_to_fc_behavior_uniform(uniform_fp_bhv, uniform_fc_bhv):
    """Test fp_to_fc for uniform behavior case."""
    np.testing.assert_allclose(fp_to_fc(uniform_fp_bhv, behavior=True), uniform_fc_bhv, atol=1e-8)


def test_fp_to_fc_behavior_pr_box(pr_box_fp, pr_box_fc_qetlab):
    """Test fp_to_fc for behavior case (PR box) using QETLAB convention."""
    np.testing.assert_allclose(fp_to_fc(pr_box_fp, behavior=True), pr_box_fc_qetlab, atol=1e-8)


def test_fp_to_fc_non_binary_exception():
    """Test fp_to_fc raises error for non-binary outcomes."""
    v_mat_non_binary = np.zeros((3, 2, 2, 2))
    with pytest.raises(ValueError, match="FP to FC conversion currently only supports binary outcomes"):
        fp_to_fc(v_mat_non_binary)
    with pytest.raises(ValueError, match="FP to FC conversion currently only supports binary outcomes"):
        fp_to_fc(v_mat_non_binary, behavior=True)


def test_fp_to_fc_behavior_zero_inputs():
    """Test fp_to_fc behavior with zero inputs (edge case for coverage)."""
    v_mat_ia0 = np.zeros((2, 2, 0, 1))
    v_mat_ib0 = np.zeros((2, 2, 1, 0))

    expected_fc_ia0 = np.array([[1.0, 0.0]])
    expected_fc_ib0 = np.array([[1.0], [0.0]])
    np.testing.assert_allclose(fp_to_fc(v_mat_ia0, behavior=True), expected_fc_ia0)
    np.testing.assert_allclose(fp_to_fc(v_mat_ib0, behavior=True), expected_fc_ib0)


def test_conversions_round_trip_functional(chsh_cg, chsh_fc, chsh_fp_func, chsh_fp_func_doc, desc_2222):
    """Test round trip conversions for functional case."""
    np.testing.assert_allclose(fc_to_cg(cg_to_fc(chsh_cg)), chsh_cg, atol=1e-8)
    np.testing.assert_allclose(cg_to_fc(fc_to_cg(chsh_fc)), chsh_fc, atol=1e-8)

    fp_from_cg_ex = cg_to_fp(chsh_cg, desc_2222)
    np.testing.assert_allclose(fp_to_cg(fp_from_cg_ex), chsh_cg, atol=1e-8)
    np.testing.assert_allclose(fp_to_fc(fc_to_fp(chsh_fc)), chsh_fc, atol=1e-8)
    np.testing.assert_allclose(cg_to_fp(fp_to_cg(chsh_fp_func_doc), desc_2222), chsh_fp_func_doc, atol=1e-8)
    np.testing.assert_allclose(fc_to_fp(fp_to_fc(chsh_fp_func)), chsh_fp_func, atol=1e-8)


def test_conversions_round_trip_behavior_cg_fc(pr_box_cg_qetlab, pr_box_fc_qetlab):
    """Test round trip conversions for behavior case where they hold (CG<->FC)."""
    fc_from_cg = cg_to_fc(pr_box_cg_qetlab, behavior=True)
    cg_rt_from_fc = fc_to_cg(fc_from_cg, behavior=True)
    np.testing.assert_allclose(cg_rt_from_fc, pr_box_cg_qetlab, atol=1e-8)

    cg_from_fc = fc_to_cg(pr_box_fc_qetlab, behavior=True)
    fc_rt_from_cg = cg_to_fc(cg_from_fc, behavior=True)
    np.testing.assert_allclose(fc_rt_from_cg, pr_box_fc_qetlab, atol=1e-8)


def test_conversions_round_trip_behavior_fc_fp_fc(pr_box_fc_qetlab):
    """Test FC -> FP -> FC round trip for behavior case."""
    fp_from_fc = fc_to_fp(pr_box_fc_qetlab, behavior=True)
    fc_rt_from_fp = fp_to_fc(fp_from_fc, behavior=True)
    np.testing.assert_allclose(fc_rt_from_fp, pr_box_fc_qetlab, atol=1e-8)


def test_conversions_round_trip_behavior_non_binary(cg_3221_bhv, fp_3221_bhv, desc_3221):
    """Test non-binary behavior round trips where possible (CG<->FP)."""
    cg_rt = fp_to_cg(cg_to_fp(cg_3221_bhv, desc_3221, behavior=True), behavior=True)
    np.testing.assert_allclose(cg_rt, cg_3221_bhv, atol=1e-8)
    fp_rt = cg_to_fp(fp_to_cg(fp_3221_bhv, behavior=True), desc_3221, behavior=True)
    np.testing.assert_allclose(fp_rt, fp_3221_bhv, atol=1e-8)

"""Unit test for Bell notation conversion functions."""
import numpy as np
import pytest

from toqito.helper.bell_notation_conversions import cg2fc, cg2fp, fc2cg, fc2fp, fp2cg, fp2fc


def test_cg2fc_bell_functional():
    """Test converting Collins-Gisin to Full Correlator notation for Bell functional."""
    cg_mat = np.array([[1, 0.5, 0.5],
                       [0.5, 0.25, 0.25],
                       [0.5, 0.25, 0.25]])

    fc_expected = np.array([[2.25  , 0.375 , 0.375 ],
                           [0.375 , 0.0625, 0.0625],
                           [0.375 , 0.0625, 0.0625]])
    fc_actual = cg2fc(cg_mat, behaviour=False)
    np.testing.assert_array_almost_equal(fc_actual, fc_expected)


def test_cg2fc_behaviour():
    """Test converting Collins-Gisin to Full Correlator notation for behaviour."""
    # This test case represents a uniform distribution p(ab|xy)=1/4
    # which should yield zero correlators.
    cg_mat = np.array([[1, 0.5, 0.5],
                       [0.5, 0.25, 0.25],
                       [0.5, 0.25, 0.25]])
    fc_expected = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
    fc_actual = cg2fc(cg_mat, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual, fc_expected)


def test_fc2cg_bell_functional():
    """Test converting Full Correlator to Collins-Gisin notation for Bell functional."""
    fc_mat = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    cg_expected = np.array([[ 3., -2., -2.],
                           [-2.,  4.,  0.],
                           [-2.,  0.,  4.]])
    cg_actual = fc2cg(fc_mat, behaviour=False)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)


def test_fc2cg_behaviour():
    """Test converting Full Correlator to Collins-Gisin notation for behaviour."""
    fc_mat = np.array([[1, -1, -1],
                       [-1, 1, 1],
                       [-1, 1, 1]])

    cg_expected = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
    cg_actual = fc2cg(fc_mat, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)


def test_cg2fp_bell_functional():
    """Test converting Collins-Gisin to Full Probability notation for Bell functional."""
    cg_mat = np.array([[1, 0.5], [0.5, 0.25]]) # ia=1, ib=1, oa=2, ob=2
    fp_expected = np.array([[[[2.25]], [[1.5]]],
                            [[[1.5]], [[1.0]]]])
    fp_actual = cg2fp(cg_mat, (2, 2), (1, 1), behaviour=False)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)


def test_cg2fp_behaviour():
    """Test converting Collins-Gisin to Full Probability notation for behaviour."""
    # Input corresponds to p(ab|xy)=0.25
    cg_mat = np.array([[1, 0.5], [0.5, 0.25]]) # ia=1, ib=1, oa=2, ob=2
    fp_expected = np.ones((2, 2, 1, 1)) * 0.25
    fp_actual = cg2fp(cg_mat, (2, 2), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)


def test_fp2cg_bell_functional():
    """Test converting Full Probability to Collins-Gisin notation for Bell functional."""
    fp_mat = np.array([[[[2.25]], [[1.5]]],
                       [[[1.5]], [[1.0]]]]) # oa=2, ob=2, ia=1, ib=1
    cg_expected = np.array([[1, 0.5], [0.5, 0.25]])
    cg_actual = fp2cg(fp_mat, behaviour=False)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)


def test_fp2cg_behaviour():
    """Test converting Full Probability to Collins-Gisin notation for behaviour."""
    # Input corresponds to p(ab|xy)=0.25
    fp_mat = np.ones((2, 2, 1, 1)) * 0.25 # oa=2, ob=2, ia=1, ib=1
    cg_expected = np.array([[1, 0.5], [0.5, 0.25]])
    cg_actual = fp2cg(fp_mat, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)


def test_fc2fp_bell_functional():
    """Test converting Full Correlator to Full Probability notation for Bell functional."""
    fc_mat = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]) # ia=2, ib=2
    fp_expected = np.ones((2, 2, 2, 2)) * 0.25
    fp_actual = fc2fp(fc_mat, behaviour=False)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)


def test_fc2fp_behaviour():
    """Test converting Full Correlator to Full Probability notation for behaviour."""
    fc_mat = np.array([[1, -1, -1],
                       [-1, 1, 1],
                       [-1, 1, 1]]) # ia=2, ib=2
    # This FC corresponds to p(11|xy)=1, others 0.
    fp_expected = np.zeros((2, 2, 2, 2))
    fp_expected[1, 1, :, :] = 1.0 # V[1,1,x,y] corresponds to p(a=1,b=1|xy)
    fp_actual = fc2fp(fc_mat, behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)


def test_fp2fc_bell_functional():
    """Test converting Full Probability to Full Correlator notation for Bell functional."""
    fp_mat = np.ones((2, 2, 2, 2)) * 0.25 # oa=2, ob=2, ia=2, ib=2
    fc_expected = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

    fc_actual = fp2fc(fp_mat, behaviour=False)
    np.testing.assert_array_almost_equal(fc_actual, fc_expected)


def test_fp2fc_behaviour():
    """Test converting Full Probability to Full Correlator notation for behaviour."""
    fp_mat = np.zeros((2, 2, 2, 2))
    fp_mat[1, 1, :, :] = 1.0 # oa=2, ob=2, ia=2, ib=2
    fc_expected = np.array([[1, -1, -1],
                           [-1, 1, 1],
                           [-1, 1, 1]])

    fc_actual = fp2fc(fp_mat, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual, fc_expected)


def test_conversion_cycle_consistency_behaviour():
    """Test that converting through all notations (behaviour=True) returns to original."""
    # Start with Collins-Gisin notation for behaviour (uniform distribution)
    cg_mat = np.array([[1, 0.5, 0.5],
                       [0.5, 0.25, 0.25],
                       [0.5, 0.25, 0.25]]) # ia=2, ib=2, oa=2, ob=2 assumed

    # Convert CG -> FC -> FP -> CG, all with behaviour=True
    fc_mat = cg2fc(cg_mat, behaviour=True)
    # fc2fp assumes oa=2, ob=2 based on FC structure. ia, ib inferred from fc_mat shape.
    fp_mat = fc2fp(fc_mat, behaviour=True)
    # fp2cg infers oa, ob, ia, ib from fp_mat shape.
    cg_final = fp2cg(fp_mat, behaviour=True)

    np.testing.assert_array_almost_equal(cg_mat, cg_final)


# Added tests for code coverage:

# Tests for oa=1 or ob=1 scenarios in cg2fp (behaviour=True)
def test_cg2fp_behaviour_oa1_ob2():
    """Test cg2fp behaviour with oa=1, ob=2."""
    # cg_mat shape (1, 1 + ib*(ob-1)) = (1, 1 + 1*(2-1)) = (1, 2)
    cg_mat = np.array([[1.0, 0.6]]) # pB(0|y=0)=0.6
    # fp shape (oa, ob, ia, ib) = (1, 2, 1, 1)
    # Expected: P(a=0,b=0|x=0,y=0)=0.6, P(a=0,b=1|x=0,y=0)=0.4
    fp_expected = np.zeros((1, 2, 1, 1))
    fp_expected[0, 0, 0, 0] = 0.6
    fp_expected[0, 1, 0, 0] = 0.4
    fp_actual = cg2fp(cg_mat, (1, 2), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)

def test_cg2fp_behaviour_oa2_ob1():
    """Test cg2fp behaviour with oa=2, ob=1."""
    # cg_mat shape (1 + ia*(oa-1), 1) = (1 + 1*(2-1), 1) = (2, 1)
    cg_mat = np.array([[1.0], [0.7]]) # pA(0|x=0)=0.7
    # fp shape (oa, ob, ia, ib) = (2, 1, 1, 1)
    fp_expected = np.zeros((2, 1, 1, 1))
    fp_expected[0, 0, 0, 0] = 0.7
    fp_expected[1, 0, 0, 0] = 0.3
    fp_actual = cg2fp(cg_mat, (2, 1), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)

def test_cg2fp_behaviour_oa1_ob1():
    """Test cg2fp behaviour with oa=1, ob=1."""
    # cg_mat shape (1, 1)
    cg_mat = np.array([[1.0]]) # Just the constant term
    fp_expected = np.array([[[[1.0]]]]) # fp shape (oa, ob, ia, ib) = (1, 1, 1, 1)
    fp_actual = cg2fp(cg_mat, (1, 1), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual, fp_expected)

# Tests for oa=1 or ob=1 scenarios in fp2cg
def test_fp2cg_behaviour_oa1_ob2():
    """Test fp2cg behaviour with oa=1, ob=2."""
    # fp shape (1, 2, 1, 1). P(a=0,b=0|x=0,y=0)=0.6, P(a=0,b=1|x=0,y=0)=0.4
    fp_mat = np.zeros((1, 2, 1, 1))
    fp_mat[0, 0, 0, 0] = 0.6
    fp_mat[0, 1, 0, 0] = 0.4
    cg_expected = np.array([[1.0, 0.6]]) # cg shape (1, 2). Expect pB(0|y=0) = P(a=0,b=0|x=0,y=0)=0.6
    cg_actual = fp2cg(fp_mat, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)

def test_fp2cg_behaviour_oa2_ob1():
    """Test fp2cg behaviour with oa=2, ob=1."""
    # fp shape (2, 1, 1, 1). P(a=0,b=0|x=0,y=0)=0.7, P(a=1,b=0|x=0,y=0)=0.3
    fp_mat = np.zeros((2, 1, 1, 1))
    fp_mat[0, 0, 0, 0] = 0.7
    fp_mat[1, 0, 0, 0] = 0.3
    cg_expected = np.array([[1.0], [0.7]]) # cg shape (2, 1). Expect pA(0|x=0)=P(a=0,b=0|x=0,y=0)=0.7
    cg_actual = fp2cg(fp_mat, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)

def test_fp2cg_behaviour_oa1_ob1():
    """Test fp2cg behaviour with oa=1, ob=1."""
    fp_mat = np.array([[[[1.0]]]]) # fp shape (1, 1, 1, 1)
    cg_expected = np.array([[1.0]]) # cg shape (1, 1)
    cg_actual = fp2cg(fp_mat, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual, cg_expected)

# Test for fc2fp error check
def test_fc2fp_zero_inputs_error():
    """Test fc2fp raises error for shape implying ia<0 or ib<0."""
    with pytest.raises(ValueError, match="Input fc_mat shape must be at least"):
        fc2fp(np.zeros((0, 1)), False) # ia = -1
    with pytest.raises(ValueError, match="Input fc_mat shape must be at least"):
        fc2fp(np.zeros((1, 0)), False) # ib = -1

# Test for fc2fp edge cases ia=0 or ib=0
def test_fc2fp_zero_inputs_functional():
    """Test fc2fp functional case with ia=0 or ib=0."""
    # ia=0, ib=0 -> fc shape (1, 1)
    fc_mat_00 = np.array([[4.0]]) # K=4 -> V(a,b,x,y)=1
    fp_expected_00 = np.ones((2, 2, 0, 0)) # Shape (2,2,0,0)
    fp_actual_00 = fc2fp(fc_mat_00, behaviour=False)
    np.testing.assert_array_almost_equal(fp_actual_00, fp_expected_00)

    # ia=1, ib=0 -> fc shape (2, 1)
    fc_mat_10 = np.array([[4.0], [0.0]]) # K=4, <A0>=0 -> V(0,b,0,y)=V(1,b,0,y)=0.5
    fp_expected_10 = np.ones((2, 2, 1, 0)) * 0.5 # Shape (2,2,1,0)
    fp_actual_10 = fc2fp(fc_mat_10, behaviour=False)
    np.testing.assert_array_almost_equal(fp_actual_10, fp_expected_10)

    # ia=0, ib=1 -> fc shape (1, 2)
    fc_mat_01 = np.array([[4.0, 0.0]]) # K=4, <B0>=0 -> V(a,0,x,0)=V(a,1,x,0)=0.5
    fp_expected_01 = np.ones((2, 2, 0, 1)) * 0.5 # Shape (2,2,0,1)
    fp_actual_01 = fc2fp(fc_mat_01, behaviour=False)
    np.testing.assert_array_almost_equal(fp_actual_01, fp_expected_01)

# Test for fp2fc non-binary error check
def test_fp2fc_non_binary_error():
    """Test fp2fc raises error if oa or ob is not 2."""
    fp_mat_3211 = np.zeros((3, 2, 1, 1))
    with pytest.raises(ValueError, match="fp2fc only works with binary outcomes"):
        fp2fc(fp_mat_3211)
    fp_mat_2311 = np.zeros((2, 3, 1, 1))
    with pytest.raises(ValueError, match="fp2fc only works with binary outcomes"):
        fp2fc(fp_mat_2311)

# Test for fp2fc behaviour with ia=0 or ib=0
def test_fp2fc_behaviour_zero_inputs():
    """Test fp2fc behaviour case with ia=0 or ib=0."""
    # ia=0, ib=1 -> fp shape (2, 2, 0, 1)
    fp_mat_01 = np.zeros((2, 2, 0, 1))
    fp_mat_01[0, 0, :, 0] = 0.6 # P(00|x,0) = 0.6
    fp_mat_01[0, 1, :, 0] = 0.4 # P(01|x,0) = 0.4
    # Expected FC: K=1, <B0> = sum_{a,x} [ P(a,0|x,0) - P(a,1|x,0) ] / ia -> 0/0 -> sets to 0
    fc_expected_01 = np.array([[1.0, 0.0]]) # fc shape (1, 2). <B0> should be 0.
    fc_actual_01 = fp2fc(fp_mat_01, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual_01, fc_expected_01)

    # ia=1, ib=0 -> fp shape (2, 2, 1, 0)
    fp_mat_10 = np.zeros((2, 2, 1, 0))
    fp_mat_10[0, 0, 0, :] = 0.7 # P(00|0,y) = 0.7
    fp_mat_10[1, 0, 0, :] = 0.3 # P(10|0,y) = 0.3
    # Expected FC: K=1, <A0> = sum_{b,y} [ P(0,b|0,y) - P(1,b|0,y) ] / ib -> 0/0 -> sets to 0
    fc_expected_10 = np.array([[1.0], [0.0]]) # fc shape (2, 1). <A0> should be 0.
    fc_actual_10 = fp2fc(fp_mat_10, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual_10, fc_expected_10)

    # ia=0, ib=0 -> fp shape (2, 2, 0, 0)
    fp_mat_00 = np.zeros((2, 2, 0, 0))
    # Both sums are 0, divisions set to 0. K=1.
    fc_expected_00 = np.array([[1.0]]) # fc shape (1, 1)
    fc_actual_00 = fp2fc(fp_mat_00, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual_00, fc_expected_00)

# Tests for shape validation errors in cg2fp (behaviour=True)
def test_cg2fp_behaviour_shape_mismatch():
    """Test cg2fp behaviour raises ValueError on incorrect cg_mat shape."""
    # Case: oa=1, ob=1, but cg_mat is not (1,1)
    with pytest.raises(ValueError, match="Expected cg_mat shape"):
        cg2fp(np.zeros((1, 2)), (1, 1), (1, 1), behaviour=True)
    # Case: oa=1, ob=2, ia=1, ib=1 but cg_mat has wrong columns
    with pytest.raises(ValueError, match="Expected cg_mat shape"):
        cg2fp(np.zeros((1, 3)), (1, 2), (1, 1), behaviour=True)
    # Case: oa=2, ob=1, ia=1, ib=1 but cg_mat has wrong rows
    with pytest.raises(ValueError, match="Expected cg_mat shape"):
        cg2fp(np.zeros((3, 1)), (2, 1), (1, 1), behaviour=True)
    # Case: oa=2, ob=2, ia=1, ib=1, but cg_mat is not (2,2)
    with pytest.raises(ValueError, match="Expected cg_mat shape"):
        cg2fp(np.zeros((2, 3)), (2, 2), (1, 1), behaviour=True)


# --- Tests for Zero Inputs / Single Outputs ---

# cg2fc (Assumes oa=2, ob=2 implicitly, test ia=0/ib=0)
def test_cg2fc_zero_inputs_behaviour():
    """Test cg2fc behaviour with ia=0 or ib=0."""
    # ia=0, ib=1 -> cg shape (1, 2)
    cg_mat_01 = np.array([[1.0, 0.6]]) # K=1, pB(0|0)=0.6
    fc_expected_01 = np.array([[1.0, 0.2]]) # K=1, <B0>=2*0.6-1=0.2
    fc_actual_01 = cg2fc(cg_mat_01, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual_01, fc_expected_01)

    # ia=1, ib=0 -> cg shape (2, 1)
    cg_mat_10 = np.array([[1.0], [0.7]]) # K=1, pA(0|0)=0.7
    fc_expected_10 = np.array([[1.0], [0.4]]) # K=1, <A0>=2*0.7-1=0.4
    fc_actual_10 = cg2fc(cg_mat_10, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual_10, fc_expected_10)

    # ia=0, ib=0 -> cg shape (1, 1)
    cg_mat_00 = np.array([[1.0]])
    fc_expected_00 = np.array([[1.0]])
    fc_actual_00 = cg2fc(cg_mat_00, behaviour=True)
    np.testing.assert_array_almost_equal(fc_actual_00, fc_expected_00)

# fc2cg (Assumes oa=2, ob=2 implicitly, test ia=0/ib=0)
def test_fc2cg_zero_inputs_behaviour():
    """Test fc2cg behaviour with ia=0 or ib=0."""
    # ia=0, ib=1 -> fc shape (1, 2)
    fc_mat_01 = np.array([[1.0, 0.2]]) # K=1, <B0>=0.2
    cg_expected_01 = np.array([[1.0, 0.6]]) # K=1, pB(0|0)=(1+0.2)/2=0.6
    cg_actual_01 = fc2cg(fc_mat_01, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual_01, cg_expected_01)

    # ia=1, ib=0 -> fc shape (2, 1)
    fc_mat_10 = np.array([[1.0], [0.4]]) # K=1, <A0>=0.4
    cg_expected_10 = np.array([[1.0], [0.7]]) # K=1, pA(0|0)=(1+0.4)/2=0.7
    cg_actual_10 = fc2cg(fc_mat_10, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual_10, cg_expected_10)

    # ia=0, ib=0 -> fc shape (1, 1)
    fc_mat_00 = np.array([[1.0]])
    cg_expected_00 = np.array([[1.0]])
    cg_actual_00 = fc2cg(fc_mat_00, behaviour=True)
    np.testing.assert_array_almost_equal(cg_actual_00, cg_expected_00)

# cg2fp (Test oa=1/ob=1 and ia=0/ib=0)
def test_cg2fp_single_output_zero_input_behaviour():
    """Test cg2fp behaviour with single outputs or zero inputs."""
    # oa=1, ob=2, ia=1, ib=1 -> cg shape (1, 2)
    cg_mat_1211 = np.array([[1.0, 0.6]]) # K=1, pB(0|0)=0.6
    fp_exp_1211 = np.zeros((1, 2, 1, 1))
    fp_exp_1211[0,0,0,0]=0.6
    fp_exp_1211[0,1,0,0]=0.4
    fp_act_1211 = cg2fp(cg_mat_1211, (1, 2), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_act_1211, fp_exp_1211)

    # oa=2, ob=1, ia=1, ib=1 -> cg shape (2, 1)
    cg_mat_2111 = np.array([[1.0], [0.7]]) # K=1, pA(0|0)=0.7
    fp_exp_2111 = np.zeros((2, 1, 1, 1))
    fp_exp_2111[0,0,0,0]=0.7
    fp_exp_2111[1,0,0,0]=0.3
    fp_act_2111 = cg2fp(cg_mat_2111, (2, 1), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_act_2111, fp_exp_2111)

    # oa=1, ob=1, ia=1, ib=1 -> cg shape (1, 1)
    cg_mat_1111 = np.array([[1.0]])
    fp_exp_1111 = np.ones((1, 1, 1, 1))
    fp_act_1111 = cg2fp(cg_mat_1111, (1, 1), (1, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_act_1111, fp_exp_1111)

    # oa=2, ob=2, ia=0, ib=1 -> cg shape (1, 2)
    cg_mat_2201 = np.array([[1.0, 0.6]]) # K=1, pB(0|0)=0.6
    # Expect P(a,b|x,y)=P(b|y) as Alice has no input choice
    fp_exp_2201 = np.zeros((2, 2, 0, 1))
    fp_exp_2201[:, 0, :, 0] = 0.6
    # It should be p(a,0|x,0)=pB(0|0)=0.6, p(a,1|x,0)=pB(1|0)=0.4
    fp_exp_2201_corr = np.zeros((2, 2, 0, 1))
    fp_exp_2201_corr[0, 0, :, 0] = 0.6 # Assume P(0,0|x,0)=pB(0|0)
    fp_exp_2201_corr[1, 0, :, 0] = 0.0 # Assume P(1,0|x,0)=0
    fp_exp_2201_corr[0, 1, :, 0] = 0.4 # Assume P(0,1|x,0)=pB(1|0)
    fp_exp_2201_corr[1, 1, :, 0] = 0.0 # Assume P(1,1|x,0)=0 - This is also arbitrary.

    # cg2fp uses P(a,b)=P(b|y).
    fp_exp_2201_fixed = np.zeros((2, 2, 0, 1))
    fp_exp_2201_fixed[:, 0, :, 0] = 0.6 # P(a,0|x,0) = pB(0|0) = 0.6
    fp_exp_2201_fixed[:, 1, :, 0] = 0.4 # P(a,1|x,0) = pB(1|0) = 0.4
    fp_act_2201 = cg2fp(cg_mat_2201, (2, 2), (0, 1), behaviour=True)
    np.testing.assert_array_almost_equal(fp_act_2201, fp_exp_2201_fixed)

    # oa=2, ob=2, ia=1, ib=0 -> cg shape (2, 1)
    cg_mat_2210 = np.array([[1.0], [0.7]]) # K=1, pA(0|0)=0.7
    # Expect P(a,b|x,y)=P(a|x) as Bob has no input choice
    fp_exp_2210 = np.zeros((2, 2, 1, 0))
    fp_exp_2210[0, :, 0, :] = 0.7 # P(0,b|0,y) = pA(0|0) = 0.7
    fp_exp_2210[1, :, 0, :] = 0.3 # P(1,b|0,y) = pA(1|0) = 0.3
    fp_act_2210 = cg2fp(cg_mat_2210, (2, 2), (1, 0), behaviour=True)
    np.testing.assert_array_almost_equal(fp_act_2210, fp_exp_2210)

# fp2cg (Test oa=1/ob=1 and ia=0/ib=0)
def test_fp2cg_single_output_zero_input_behaviour():
    """Test fp2cg behaviour with single outputs or zero inputs."""
    # oa=1, ob=2, ia=1, ib=1 -> fp shape (1, 2, 1, 1)
    fp_mat_1211 = np.zeros((1, 2, 1, 1))
    fp_mat_1211[0,0,0,0]=0.6
    fp_mat_1211[0,1,0,0]=0.4
    cg_exp_1211 = np.array([[1.0, 0.6]]) # K=1, pB(0|0)=sum_a fp(a,0,x=0,0)=fp(0,0,0,0)=0.6
    cg_act_1211 = fp2cg(fp_mat_1211, behaviour=True)
    np.testing.assert_array_almost_equal(cg_act_1211, cg_exp_1211)

    # oa=2, ob=1, ia=1, ib=1 -> fp shape (2, 1, 1, 1)
    fp_mat_2111 = np.zeros((2, 1, 1, 1))
    fp_mat_2111[0,0,0,0]=0.7
    fp_mat_2111[1,0,0,0]=0.3
    cg_exp_2111 = np.array([[1.0], [0.7]]) # K=1, pA(0|0)=sum_b fp(0,b,0,y=0)=fp(0,0,0,0)=0.7
    cg_act_2111 = fp2cg(fp_mat_2111, behaviour=True)
    np.testing.assert_array_almost_equal(cg_act_2111, cg_exp_2111)

    # oa=1, ob=1, ia=1, ib=1 -> fp shape (1, 1, 1, 1)
    fp_mat_1111 = np.ones((1, 1, 1, 1))
    cg_exp_1111 = np.array([[1.0]])
    cg_act_1111 = fp2cg(fp_mat_1111, behaviour=True)
    np.testing.assert_array_almost_equal(cg_act_1111, cg_exp_1111)

    # oa=2, ob=2, ia=0, ib=1 -> fp shape (2, 2, 0, 1)
    fp_mat_2201 = np.zeros((2, 2, 0, 1))
    fp_mat_2201[0, 0, :, 0] = 0.6
    fp_mat_2201[1, 0, :, 0] = 0.0 # p(a,0|x,0)
    fp_mat_2201[0, 1, :, 0] = 0.4
    fp_mat_2201[1, 1, :, 0] = 0.0 # p(a,1|x,0)
    # K=1, pA not calculated (ia=0). pB(0|0)=sum_a fp(a,0,x=0,0)=0.6+0.0=0.6
    cg_exp_2201 = np.array([[1.0, 0.0]])
    cg_act_2201 = fp2cg(fp_mat_2201, behaviour=True)
    np.testing.assert_array_almost_equal(cg_act_2201, cg_exp_2201)

    # oa=2, ob=2, ia=1, ib=0 -> fp shape (2, 2, 1, 0)
    fp_mat_2210 = np.zeros((2, 2, 1, 0))
    fp_mat_2210[0, 0, 0, :] = 0.7
    fp_mat_2210[0, 1, 0, :] = 0.0 # p(0,b|0,y)
    fp_mat_2210[1, 0, 0, :] = 0.3
    fp_mat_2210[1, 1, 0, :] = 0.0 # p(1,b|0,y)
    # K=1, pB not calculated (ib=0). pA(0|0)=sum_b fp(0,b,0,y=0)=0.7+0.0=0.7
    cg_exp_2210 = np.array([[1.0], [0.0]])
    cg_act_2210 = fp2cg(fp_mat_2210, behaviour=True)
    np.testing.assert_array_almost_equal(cg_act_2210, cg_exp_2210)

def test_fc2fp_zero_inputs_behaviour():
    """Test fc2fp behaviour case with ia=0 or ib=0."""
    # ia=1, ib=0 -> fc shape (2, 1)
    fc_mat_10 = np.array([[1.0], [0.4]]) # K=1, <A0>=0.4
    # Expected FP: P(0,0)=(1+ax)/4, P(0,1)=(1+ax)/4, P(1,0)=(1-ax)/4, P(1,1)=(1-ax)/4
    fp_expected_10 = np.zeros((2, 2, 1, 0))
    fp_expected_10[0, :, 0, :] = (1 + 0.4) / 4 # 0.35
    fp_expected_10[1, :, 0, :] = (1 - 0.4) / 4 # 0.15
    fp_actual_10 = fc2fp(fc_mat_10, behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual_10, fp_expected_10)

    # ia=0, ib=1 -> fc shape (1, 2)
    fc_mat_01 = np.array([[1.0, 0.2]]) # K=1, <B0>=0.2
    # Expected FP: P(0,0)=(1+by)/4, P(0,1)=(1-by)/4, P(1,0)=(1+by)/4, P(1,1)=(1-by)/4
    fp_expected_01 = np.zeros((2, 2, 0, 1))
    fp_expected_01[:, 0, :, 0] = (1 + 0.2) / 4 # 0.3
    fp_expected_01[:, 1, :, 0] = (1 - 0.2) / 4 # 0.2
    fp_actual_01 = fc2fp(fc_mat_01, behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual_01, fp_expected_01)

    # ia=0, ib=0 -> fc shape (1, 1)
    fc_mat_00 = np.array([[1.0]]) # K=1
    fp_expected_00 = np.ones((2, 2, 0, 0)) * 0.25 # Uniform
    fp_actual_00 = fc2fp(fc_mat_00, behaviour=True)
    np.testing.assert_array_almost_equal(fp_actual_00, fp_expected_00)

# Test the cg2fp functional case branches for ia/ib=0
def test_cg2fp_zero_inputs_functional():
     """Test cg2fp functional case with ia=0 or ib=0."""
     # ia=1, ib=0 -> cg shape (2, 1)
     cg_mat_10 = np.array([[0.0], [1.0]]) # K=0, A-marginal=1

     fp_exp_10 = np.zeros((2, 2, 1, 0)) # Shape (2, 2, 1, 0)

     fp_act_10 = cg2fp(cg_mat_10, (2, 2), (1, 0), behaviour=False)
     np.testing.assert_array_almost_equal(fp_act_10, fp_exp_10)

     # ia=0, ib=1 -> cg shape (1, 2)
     cg_mat_01 = np.array([[0.0, 1.0]]) # K=0, B-marginal=1
     fp_exp_01 = np.zeros((2, 2, 0, 1)) # Shape (2, 2, 0, 1)

     fp_act_01 = cg2fp(cg_mat_01, (2, 2), (0, 1), behaviour=False)
     np.testing.assert_array_almost_equal(fp_act_01, fp_exp_01)

     # ia=0, ib=0 -> cg shape (1, 1)
     cg_mat_00 = np.array([[0.0]])
     fp_exp_00 = np.zeros((2, 2, 0, 0))
     fp_act_00 = cg2fp(cg_mat_00, (2, 2), (0, 0), behaviour=False)
     np.testing.assert_array_almost_equal(fp_act_00, fp_exp_00)


# Test fp2cg functional case branches for ia/ib=0
def test_fp2cg_zero_inputs_functional():
    """Test fp2cg functional case with ia=0 or ib=0."""
    # ia=1, ib=0 -> fp shape (2, 2, 1, 0)
    fp_mat_10 = np.zeros((2, 2, 1, 0))
    fp_mat_10[0, 1, 0, :] = 1.0 # V(0, 1, 0, y) = 1
    fp_mat_10[1, 1, 0, :] = -1.0 # V(1, 1, 0, y) = -1

    cg_exp_10 = np.zeros((2, 1)) # Shape (2, 1)

    cg_act_10 = fp2cg(fp_mat_10, behaviour=False)
    np.testing.assert_array_almost_equal(cg_act_10, cg_exp_10)

    # ia=0, ib=1 -> fp shape (2, 2, 0, 1)
    fp_mat_01 = np.zeros((2, 2, 0, 1))
    fp_mat_01[1, 0, :, 0] = 1.0
    fp_mat_01[1, 1, :, 0] = -1.0
    cg_exp_01 = np.zeros((1, 2))

    # The sum should be over existing indices. sum(fp_mat[1,1,:,:]) -> sum(-1) = -1
    cg_exp_01[0, 0] = 0.0

    cg_act_01 = fp2cg(fp_mat_01, behaviour=False)
    np.testing.assert_array_almost_equal(cg_act_01, cg_exp_01)

    # ia=0, ib=0 -> fp shape (2, 2, 0, 0)
    fp_mat_00 = np.zeros((2, 2, 0, 0))
    cg_exp_00 = np.zeros((1, 1)) # Shape (1, 1)
    cg_act_00 = fp2cg(fp_mat_00, behaviour=False)
    np.testing.assert_array_almost_equal(cg_act_00, cg_exp_00)


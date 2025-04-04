"""Unit test for Bell notation conversion functions."""
import numpy as np

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


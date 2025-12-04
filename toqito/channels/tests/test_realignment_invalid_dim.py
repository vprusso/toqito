import pytest
import numpy as np
from toqito.channels.realignment import realignment

def test_realignment_invalid_dim_zero():
    mat = np.eye(4)
    with pytest.raises(ValueError):
        realignment(mat, dim=0)

def test_realignment_invalid_dim_negative_in_list():
    mat = np.eye(4)
    with pytest.raises(ValueError):
        realignment(mat, dim=[1, -2])

def test_realignment_invalid_dim_non_int_in_list():
    mat = np.eye(4)
    with pytest.raises(TypeError):
        realignment(mat, dim=[1, 2.5])

def test_realignment_invalid_dim_wrong_type():
    mat = np.eye(4)
    with pytest.raises(TypeError):
        realignment(mat, dim="a")

"""Compute the dual of a map"""
from typing import List, Union
import numpy as np

def dual_channel(phi: Union[np.ndarray, List[List[np.ndarray]]]) -> Union[np.ndarray, List[List[np.ndarray]]]:
	r"""
	TODO: docstring
	"""
	# If phi is a list, assume it contains couples of Kraus operators, and take the Hermitian conjugate
	if isinstance(phi, list):
		return [[a.conj().T for a in x] for x in phi]


    # If phi is a ndarray, assume it is a Choi matrix
	if isnstance(phi, np.ndarray):
        # swap
		pass
    raise ValueError(
        "Invalid: The variable `phi_op` must either be a list of "
        "Kraus operators or as a Choi matrix."
    )

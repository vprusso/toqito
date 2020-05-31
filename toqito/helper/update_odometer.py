"""Update odometer."""
from typing import List, Union
import numpy as np


def update_odometer(
    old_ind: Union[List[int], np.ndarray], upper_lim: Union[List[int], np.ndarray]
) -> List[int]:
    r"""
    Increase a vector as odometer.

    Increases the last entry of the vector `old_ind` by 1, unless that would
    make it larger than the last entry of the vector `upper_lim`. In this case,
    it sets the last entry to 0 and instead increases the second-last entry of
    `old_ind`, unless that would make it larger than the second-last entry of
    `upper_lim`. In this case, it sets the second-last entry to 0 and instead
    increases the third-last entry of `old_ind` (and so on; it works like an
    odometer).

    This function is useful when you want to have k nested loops, but k isn't
    specified beforehand. For example, instead of looping over i and j going
    from 1 to 3, you could loop over a single variable going from 1 to 3^2 and
    set [i, j] = update_odometer([i, j], [3, 3]) at each step within the loop.

    This function is adapted from QETLAB [QETOD]_.

    Examples
    ==========

    >>> from toqito.helper import update_odometer
    >>> import numpy as np
    >>> vec = np.array([0, 0])
    >>> upper_lim = np.array([3, 2])
    >>> for j in range(0, np.prod(upper_lim)-1):
    >>> vec = update_odometer(vec, upper_lim)
    >>> vec
    [0 1]
    [1 0]
    [1 1]
    [2 0]
    [2 1]

    References
    ==========
    .. [QETOD] QETLAB: Update odometer function.
        http://www.qetlab.com/update_odometer

    :param old_ind: The initial vector.
    :param upper_lim: The upper limit on which to increase the odometer to.
    :return: The updated vector.
    """
    ind_len = len(old_ind)
    new_ind = old_ind[:]

    # Start by increasing the last index by 1.
    if len(new_ind) > 0:
        new_ind[-1] = new_ind[-1] + 1

    # Increment the "odometer": Repeatedly set each digit to 0 if it is too high
    # and carry the addition to the left until we hit a digit that is not too
    # high.
    for j in range(ind_len, 0, -1):
        # If we have hit the upper limit in this entry, move onto the next
        # entry.
        if new_ind[j - 1] >= upper_lim[j - 1]:
            new_ind[j - 1] = 0
            if j >= 2:
                new_ind[j - 2] = new_ind[j - 2] + 1
            else:
                # We are at the left end of the vector, so just stop.
                return new_ind
        else:
            # Always return if the odometer doesn't turn over.
            return new_ind
    return new_ind

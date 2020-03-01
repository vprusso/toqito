"""Compute all distinct permutations of a given vector."""
from typing import List
from dataclasses import dataclass


@dataclass
class UniqueElement:
    """Class for unique elements to keep track of occurrences."""

    value: int
    occurrences: int


def unique_perms(elements: List[int]):
    """
    Determine the number of unique permutations of a list.

    :param elements:
    :return:
    """
    elem_set = set(elements)
    list_unique = [UniqueElement(value=i, occurrences=elements.count(i)) for i in elem_set]
    len_elems = len(elements)

    return perm_unique_helper(list_unique, [0]*len_elems, len_elems-1)


def perm_unique_helper(list_unique: List[UniqueElement],
                       result_list: List[int],
                       elem_d: int):
    """
    Provide helper function for unique_perms.

    :param list_unique:
    :param result_list:
    :param elem_d:
    :return:
    """
    if elem_d < 0:
        yield tuple(result_list)
    else:
        for i in list_unique:
            if i.occurrences > 0:
                result_list[elem_d] = i.value
                i.occurrences -= 1
                for g_perm in perm_unique_helper(list_unique,
                                                 result_list,
                                                 elem_d-1):
                    yield g_perm
                i.occurrences += 1

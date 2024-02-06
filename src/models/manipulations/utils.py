from typing import List


def search_by_prefix(
    query: str,
    target_list: List[str],
):
    """
    Args:
        query: a string to match with
        target_list: a list of strings to be matched
    Returns:
        result: a string from target_list that has the longest common prefix with query. On tie, the shortest string is returned.
    """

    def _common_prefix_length(str_a, str_b):
        idx = 0
        for idx in range(min(len(str_a), len(str_b))):
            if str_a[idx] != str_b[idx]:
                return idx
        return min(len(str_a), len(str_b))

    curr_best_tuple, result = None, None
    for target in target_list:
        value_tuple = _common_prefix_length(query, target), -len(target)
        if curr_best_tuple is None or value_tuple > curr_best_tuple:
            curr_best_tuple, result = value_tuple, target

    return result

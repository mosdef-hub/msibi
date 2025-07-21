import re

import numpy as np


def calc_similarity(arr1, arr2):
    """F-fit score used in MSIBI paper."""
    f_fit = np.sum(np.absolute(arr1 - arr2))
    f_fit /= np.sum((np.absolute(arr1) + np.absolute(arr2)))
    return 1.0 - f_fit


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'."""
    idx = np.abs(array - target).argmin()
    return idx, array[idx]


def _atoi(text):
    """Convert string digit to int."""
    return int(text) if text.isdigit() else text


def natural_sort(text):
    """Break apart a string containing letters and digits."""
    return [_atoi(a) for a in re.split(r"(\d+)", text)]

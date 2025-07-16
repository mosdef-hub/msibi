import numpy as np

from msibi.utils.error_calculation import calc_similarity
from msibi.utils.general import find_nearest


def test_calc_similarity():
    a = np.arange(0.0, 5.0, 0.05)
    b = np.arange(0.0, 5.0, 0.05)
    assert calc_similarity(a, b) == 1.0
    b *= -1
    assert calc_similarity(a, b) == 0.0
    arr1 = np.random.random(10)
    arr2 = np.random.random(10)
    assert calc_similarity(arr1, arr2) == calc_similarity(arr2, arr1)


def test_find_nearest():
    a = np.arange(10)
    idx, nearest = find_nearest(a, 2.1)
    assert idx == 2
    assert nearest == 2

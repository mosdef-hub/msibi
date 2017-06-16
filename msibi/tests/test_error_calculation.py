import numpy as np

from msibi.utils.error_calculation import calc_similarity


def test_calc_similarity():
    arr1 = np.ones(10)
    arr2 = np.ones(10)
    f_fit = calc_similarity(arr1, arr2)
    assert f_fit == 1.0

    arr2 = np.zeros(10)
    f_fit = calc_similarity(arr1, arr2)
    assert f_fit == 0.0

    arr1 = np.random.random(10)
    arr2 = np.random.random(10)
    assert calc_similarity(arr1, arr2) == calc_similarity(arr2, arr1)

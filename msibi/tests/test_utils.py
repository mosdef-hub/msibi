import numpy as np
import pytest

from msibi.utils.error_calculation import calc_similarity
from msibi.utils.general import find_nearest
from msibi.utils.smoothing import savitzky_golay


def test_calc_similarity():
    a = np.arange(0.0, 5.0, 0.05)
    b = np.arange(0.0, 5.0, 0.05)
    assert calc_similarity(a, b) == 1.0
    b *= -1
    assert calc_similarity(a, b) == 0.0


def test_find_nearest():
    a = np.arange(10)
    idx, nearest = find_nearest(a, 2.1)
    assert idx == 2
    assert nearest == 2


def test_savitzky_golay():
    x = np.arange(0, 1, 0.01)
    y = 2 * x + 1
    y2 = savitzky_golay(y, 3, 1)
    assert y.shape == y2.shape
    assert np.allclose(y, y2)

    y = x ** 3.0
    y2 = savitzky_golay(y, 3, 1)
    assert calc_similarity(y, y2) > 0.99

    with pytest.raises(ValueError):
        y2 = savitzky_golay(y, 3.1, 1)
    with pytest.raises(TypeError):
        y2 = savitzky_golay(y, 2, 1)
    with pytest.raises(TypeError):
        y2 = savitzky_golay(y, 4, 1)
    with pytest.raises(TypeError):
        y2 = savitzky_golay(y, 3, 3)
    with pytest.raises(TypeError):
        y2 = savitzky_golay(y, 3, 2)

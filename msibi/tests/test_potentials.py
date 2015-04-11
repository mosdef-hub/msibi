import pytest
import numpy as np

from msibi.potentials import tail_correction, mie, alpha_array


def test_tail_correction():
    dr = 0.05
    r = np.arange(0, 2.5, dr)
    V = mie(r, 1, 1)

    smooth_V = tail_correction(r, V, r_switch=2.25)
    assert smooth_V[-1] == 0.0

def test_calc_alpha_array():
    alpha0 = 1.0
    dr = 0.1
    r = np.arange(0, 2.5, dr)
    form = 'linear'
    alpha = alpha_array(alpha0, r, form)
    assert alpha[0] == alpha0
    assert alpha[-1] == 0.0

    form = 'linearlkjlkdasfj'
    with pytest.raises(ValueError):
        alpha = alpha_array(alpha0, r, form)


if __name__ == "__main__":
    test_tail_correction()


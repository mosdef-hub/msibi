import numpy as np

from msibi.utils.potentials import alpha_array

# def test_tail_correction():
#    dr = 0.05
#    r = np.arange(0, 2.5, dr)
#    V = mie(r, 1, 1, m=12, n=6)
#    smooth_V = _pair_tail_correction(r, V, r_switch=2.25)
#    assert smooth_V[-1] == 0.0


def test_calc_alpha_array():
    alpha0 = 1.0
    dr = 0.1
    r = np.arange(dr, 2.5 + dr, dr)
    alpha = alpha_array(alpha0=alpha0, pot_r=r, dr=dr, form="linear")
    assert len(alpha) == len(r)
    assert alpha[0] == alpha0
    assert alpha[-1] == 0.0

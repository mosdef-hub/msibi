import numpy as np

from msibi.potentials import tail_correction, mie


def test_tail_correction():
    dr = 0.05
    r = np.arange(0, 2.5, dr)
    V = mie(r, 1, 1)

    smooth_r, smooth_V, smooth_F = tail_correction(r, dr, V)
    assert smooth_r.shape == (len(r) + 1,)
    assert smooth_r.shape == smooth_V.shape
    assert smooth_r.shape == smooth_F.shape
    assert smooth_V[-1] == 0.0


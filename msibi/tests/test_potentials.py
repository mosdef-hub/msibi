import numpy as np

from msibi.potentials import tail_correction, mie


def test_tail_correction():
    dr = 0.05
    r = np.arange(0, 2.5, dr)
    V = mie(r, 1, 1)

    smooth_r, smooth_V = tail_correction(r, dr, V, r_switch=2.25)
    assert smooth_V[-1] == 0.0

if __name__ == "__main__":
    test_tail_correction()


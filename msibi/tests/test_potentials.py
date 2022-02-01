import numpy as np
import pytest

from msibi.potentials import alpha_array, tail_correction, head_correction


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
    form = "linear"
    alpha = alpha_array(alpha0, r, form)
    assert alpha[0] == alpha0
    assert alpha[-1] == 0.0

    form = "margaret-thatcher"
    with pytest.raises(ValueError):
        alpha = alpha_array(alpha0, r, form)


def test_head_correction():
    dr = 0.05
    cutoff = 8
    r = np.arange(0, 2.5, dr)
    V_prev = mie(r, 1, 1)
    V = mie(r, 2, 1)
    V[:cutoff] = np.inf

    linear_V = head_correction(r, np.copy(V), V_prev, "linear")
    assert not np.isnan(linear_V).any() and not np.isinf(linear_V).any()
    assert all(linear_V[cutoff:] == V[cutoff:])

    exp_V = head_correction(r, np.copy(V), V_prev, "exponential")
    assert not np.isnan(exp_V).any() and not np.isinf(exp_V).any()
    assert all(exp_V[cutoff:] == V[cutoff:])
    assert exp_V[0] > linear_V[0]

import numpy as np

from msibi.utils.corrections import (
    bonded_corrections,
    exponential,
    harmonic,
    linear,
    pair_corrections,
)
from msibi.utils.potentials import mie


def generate_parabolic_potential(
    x0=0, x_range=(0, 4), num_points=100, noise_level=0.05
):
    """Generate parabolic potential with optional noise added."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    V_x = (x_values - x0) ** 2
    noise = np.random.normal(0, noise_level, len(x_values))
    V_x_noisy = V_x + noise
    return x_values, V_x_noisy


def generate_lj_potential(
    x_range=(0.1, 4), num_points=100, noise_level=0.05, epsilon=1, sigma=1
):
    """Generate 12-6 LJ potential with optional noise added."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    V_x = mie(r=x_values, epsilon=epsilon, sigma=sigma, m=12, n=6)
    noise = np.random.normal(0, noise_level, len(x_values))
    V_x_noisy = V_x + noise
    return x_values, V_x_noisy


def test_harmonic_bonded_correction():
    """Make sure corrections recover original harmonic potential."""
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0)
    V_missing = np.copy(V)
    V_missing[0:15] = np.inf
    V_missing[-15:] = np.inf
    V_corrected, head_start, tail_start, real_indices = bonded_corrections(
        x=x,
        V=V_missing,
        fit_window_size=15,
        head_correction_func=harmonic,
        tail_correction_func=harmonic,
        smoothing_order=None,
        smoothing_window=None,
    )
    assert np.allclose(V, V_corrected, atol=1e-5)
    assert np.array_equal(real_indices, np.arange(15, 85))
    assert head_start == 15
    assert tail_start == 85


def test_linear_bonded_correction():
    """Make sure non-harmonic correction gives different potential."""
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0)
    V_missing = np.copy(V)
    V_missing[0:15] = np.inf
    V_missing[-15:] = np.inf
    V_corrected, head_start, tail_start, real_indices = bonded_corrections(
        x=x,
        V=V_missing,
        fit_window_size=15,
        head_correction_func=linear,
        tail_correction_func=linear,
        smoothing_order=None,
        smoothing_window=None,
    )
    assert not np.allclose(V, V_corrected, atol=1e-5)
    # Make sure correction gives increasing V(x) with decreasing x
    for y in V_corrected[0:head_start]:
        assert y > np.max(V[real_indices])
    # Make sure correction gives increasing V(x) with increasing x
    for y in V_corrected[tail_start:]:
        assert y > np.max(V[real_indices])
    assert np.array_equal(real_indices, np.arange(15, 85))
    assert head_start == 15
    assert tail_start == 85


def test_exponential_bonded_correction():
    """Make sure non-harmonic correction gives different potential."""
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0)
    V_missing = np.copy(V)
    V_missing[0:15] = np.inf
    V_missing[-15:] = np.inf
    V_corrected, head_start, tail_start, real_indices = bonded_corrections(
        x=x,
        V=V_missing,
        fit_window_size=15,
        head_correction_func=exponential,
        tail_correction_func=exponential,
        smoothing_order=None,
        smoothing_window=None,
    )
    assert not np.allclose(V, V_corrected, atol=1e-5)
    # Make sure correction gives increasing V(x) with decreasing x
    for y in V_corrected[0:head_start]:
        assert y > np.max(V[real_indices])
    # Make sure correction gives increasing V(x) with increasing x
    for y in V_corrected[tail_start:]:
        assert y > np.max(V[real_indices])
    assert np.array_equal(real_indices, np.arange(15, 85))
    assert head_start == 15
    assert tail_start == 85


def test_pair_tail_corrections():
    x, V = generate_lj_potential(noise_level=0)
    V_corrected, head_start, idx_switch, real_indices = pair_corrections(
        x=x,
        V=V,
        fit_window_size=10,
        r_switch=2,
        smoothing_window=None,
        smoothing_order=None,
        head_correction_func=exponential,
    )
    assert V_corrected[-1] == 0
    for v1, v2 in zip(V[idx_switch + 1 :], V_corrected[idx_switch + 1 :]):
        assert np.abs(v1) > np.abs(v2)


def test_pair_no_tail_corrections():
    x, V = generate_lj_potential(noise_level=0)
    V_corrected, head_start, idx_switch, real_indices = pair_corrections(
        x=x,
        V=V,
        fit_window_size=10,
        r_switch=None,
        smoothing_window=None,
        smoothing_order=None,
        head_correction_func=exponential,
    )
    assert np.array_equal(V, V_corrected)

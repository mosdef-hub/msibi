import numpy as np
import pytest

from msibi.utils.corrections import (
    _anchored_predict,
    _select_window,
    bonded_corrections,
    exponential,
    harmonic,
    linear,
    pair_corrections,
)
from msibi.utils.potentials import mie


def generate_parabolic_potential(
    x0=0, x_range=(0, 4), num_points=100, noise_level=0.05, seed=None
):
    """Generate parabolic potential with optional noise added.

    Pass a seed to get a reproducible noise draw.
    """
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    V_x = (x_values - x0) ** 2
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, len(x_values))
    V_x_noisy = V_x + noise
    return x_values, V_x_noisy


def generate_lj_potential(
    x_range=(0.1, 4), num_points=100, noise_level=0.05, epsilon=1, sigma=1, seed=None
):
    """Generate 12-6 LJ potential with optional noise added.

    Pass a seed to get a reproducible noise draw.
    """
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    V_x = mie(r=x_values, epsilon=epsilon, sigma=sigma, m=12, n=6)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, len(x_values))
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
        maxfev=1000,
        smoothing_order=None,
        smoothing_window=None,
    )
    assert np.allclose(V, V_corrected, atol=1e-5)
    assert np.array_equal(real_indices, np.arange(15, 85))
    assert head_start == 15
    assert tail_start == 85


def test_undefined_error():
    """Catch error when large region of undefined values exist within defined potential range."""
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0)
    V_missing = np.copy(V)
    V_missing[15:25] = np.inf
    with pytest.raises(RuntimeError):
        bonded_corrections(
            x=x,
            V=V_missing,
            fit_window_size=15,
            head_correction_func=harmonic,
            tail_correction_func=harmonic,
            maxfev=1000,
            smoothing_order=None,
            smoothing_window=None,
        )


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
        maxfev=1000,
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
        maxfev=1000,
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
    V_missing = np.copy(V)
    V_missing[0:15] = np.inf
    V_corrected, head_start, idx_switch, _real_indices = pair_corrections(
        x=x,
        V=V_missing,
        fit_window_size=10,
        r_switch=2,
        maxfev=3000,
        smoothing_window=None,
        smoothing_order=None,
        head_correction_func=exponential,
    )
    assert V_corrected[-1] == 0
    assert head_start == 15
    for v1, v2 in zip(V[idx_switch + 1 :], V_corrected[idx_switch + 1 :]):
        assert np.abs(v1) > np.abs(v2)


def test_pair_no_tail_corrections():
    x, V = generate_lj_potential(noise_level=0)
    V_missing = np.copy(V)
    V_missing[0:15] = np.inf
    V_corrected, head_start, _idx_switch, _real_indices = pair_corrections(
        x=x,
        V=V_missing,
        fit_window_size=10,
        r_switch=None,
        smoothing_window=None,
        smoothing_order=None,
        maxfev=3000,
        head_correction_func=exponential,
    )
    assert not np.allclose(V, V_corrected, atol=1e-3)
    assert head_start == 15


@pytest.mark.parametrize("side", ["head", "tail"])
@pytest.mark.parametrize("form", [harmonic, exponential, linear])
def test_anchored_prediction_is_c1_continuous(side, form):
    """The extrapolation must join the real data with matching value and slope.

    For any correction form, window and noise level, the filled region leaves the
    seam at the value and first derivative of the real data, so the force is
    continuous there. Checked directly on _anchored_predict, where the match is
    exact rather than approximate.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.5, 4.0, 200)
    # An anharmonic well, so the harmonic and exponential forms are misspecified.
    v = 8.0 * (x - 2.0) ** 4 + 4.0 * (x - 2.0) ** 2 + rng.normal(0, 0.02, x.size)

    x_b = x[0] if side == "head" else x[-1]
    step = 1e-6
    into_gap = x_b - step if side == "head" else x_b + step
    x_probe = np.array([x_b, into_gap])
    pred = _anchored_predict(x, v, x_probe, form, 5000, side, 12)

    # Value continuity: the curve passes exactly through the boundary datum.
    v_b = v[0] if side == "head" else v[-1]
    assert np.isclose(pred[0], v_b, atol=1e-9)

    # Slope continuity: the slope leaving the boundary equals the data's local
    # slope there (the noise-averaged boundary slope estimate).
    if side == "head":
        u, vv = x[:12] - x_b, v[:12]
    else:
        u, vv = x[-12:] - x_b, v[-12:]
    s_data = np.polyval(np.polyder(np.polyfit(u, vv, 2)), 0.0)
    s_pred = (pred[1] - pred[0]) / (into_gap - x_b)
    assert np.isclose(s_pred, s_data, rtol=1e-4)


def test_auto_window_sweep_fills_gap():
    """fit_window_size=None chooses the window automatically and fills the gap."""
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0.02, seed=0)
    V_missing = np.copy(V)
    V_missing[0:15] = np.inf
    V_missing[-15:] = np.inf
    V_corrected, head_start, tail_start, _real_indices = bonded_corrections(
        x=x,
        V=V_missing,
        fit_window_size=None,
        head_correction_func=harmonic,
        tail_correction_func=harmonic,
        maxfev=3000,
        smoothing_order=None,
        smoothing_window=None,
    )
    # The gap is filled with finite values and the head/tail rise above the well.
    assert np.all(np.isfinite(V_corrected))
    assert head_start == 15
    assert tail_start == 85
    assert np.all(V_corrected[:head_start] > np.min(V_corrected))
    assert np.all(V_corrected[tail_start:] > np.min(V_corrected))


def test_select_window_returns_window_in_range():
    """The sweep returns an integer window inside the searched bounds."""
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0.02, seed=0)
    real = np.arange(15, 85)
    w = _select_window(x[real], V[real], x[:15], harmonic, 3000, side="head")
    assert isinstance(w, (int, np.integer))
    assert 4 <= w <= 25


def test_stability_tie_break_overrides_rmse_argmin():
    """On noisy data the tie-break picks a steadier window than the RMSE argmin.

    Fails if selection is reduced to the raw argmin.
    """
    x, V = generate_parabolic_potential(x0=2, x_range=(0, 4), noise_level=0.05, seed=4)
    real = np.arange(15, 85)
    x_real, v_real, x_missing = x[real], V[real], x[:15]
    w, scores = _select_window(
        x_real,
        v_real,
        x_missing,
        harmonic,
        3000,
        side="head",
        return_scores=True,
    )
    rmse = {k: s["rmse"] for k, s in scores.items()}
    instability = {k: s["instability"] for k, s in scores.items()}
    argmin = min(rmse, key=rmse.get)

    tol = rmse[argmin] * 1.25 + 1e-12 * np.max(np.abs(v_real))
    tied = [k for k, r in rmse.items() if r <= tol]

    # The chosen window is steadier than the argmin and still tied on RMSE.
    assert w != argmin
    assert instability[w] < instability[argmin]
    assert rmse[w] <= tol
    assert w == min(tied, key=lambda k: instability[k])

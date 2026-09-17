from collections.abc import Callable

import more_itertools as mit
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from msibi.utils.general import find_nearest

bad_fit_error_msg = """\
There was an error fitting the existing potential against the head/tail correction function.
This could indicate poor sampling, or a poor choice of fitting parameters.

Try adjusting parameters that impact sampling such as gsd_period in msibi.MSIBI,
sampling_stride and n_frames in msibi.State, and n_steps in MSIBI.run_optimization.
You might adjust the fitting and smoothing related parameters.
See smoothing_window, correction_fit_window, smoothing_order, maxfev in msibi.Force.
"""


def harmonic(x: np.ndarray, x0: float, k: float):
    """Used as the default correction form for bonded forces.

        :math:`V(x) = 0.5*k(x - x0)^2`

    .. note::

        This is used by default for msibi.force.Bond, msibi.force.Angle
        and msibi.force.Dihedral.

    """
    return 0.5 * k * (x - x0) ** 2


def exponential(x: np.ndarray, A: float, B: float):
    """Used as the default head correction for non-bonded pair potentials.

    :math:`V(x) = A*exp(-Bx)`

    """
    return A * np.exp(-B * x)


def linear(x: np.ndarray, m: float, b: float):
    """Functional form that can be used for head or tail corrections.

    :math:`V(x) = mx + b`


    """
    return x * m + b


def bonded_corrections(
    x: np.ndarray,
    V: np.ndarray,
    smoothing_window: int,
    smoothing_order: int,
    fit_window_size: int | None,
    maxfev: int,
    head_correction_func: Callable,
    tail_correction_func: Callable,
):
    """The default correction method for bonded forces.

    If fit_window_size is None, the fit window for each side is chosen
    automatically by _select_window (a walk-forward window sweep) instead of
    using a single user-supplied window.

    Parameters
    ----------
    x : np.ndarray
        The x values of the force.
    V : np.ndarray
        The y values of the force.
    smoothing_window : int
        Window size to use in scipy.signal.savgol_fiter
        to smooth V before fitting.
    smoothing_order : int
        Polynomial order to use in scipy.signal.savgol_fiter
        to smooth V before fitting.
    head_correction_func : Callable
        Functional form to use in fitting the head (left) side
        of the potential.
    tail_correction_func : Callable
        Functional form to use in fitting the tail (right) side
        of the potential.
    """
    V = np.copy(V)
    real_indices = _get_real_indices(V)
    v_real = np.copy(V[real_indices])
    if len(v_real) == len(V):
        print("No regions to account for corrections")
        return V, None, None, None
    x_real = np.copy(x[real_indices])
    head_start = real_indices[0]
    tail_start = real_indices[-1]
    # If the info is there, apply some smoothing before using SciPy curve_fit
    # The smoothed real portion of the potential isn't retained here
    # That is handled seaprately and performed on the potential with head & tail corrections included
    if all([smoothing_window, smoothing_order]):
        if len(v_real) < 2 * smoothing_window:
            mode = "nearest"
        else:
            mode = "interp"

        v_real = savgol_filter(
            x=v_real,
            window_length=smoothing_window,
            polyorder=smoothing_order,
            mode=mode,
        )

    # Choose per-side fit windows. When fit_window_size is None, sweep candidate
    # windows and keep the one that best extrapolates onto held-out real data.
    x_head_missing = x[:head_start]
    x_tail_missing = x[tail_start + 1 :]
    if fit_window_size is None:
        head_window = _select_window(
            x_real,
            v_real,
            x_head_missing,
            head_correction_func,
            maxfev,
            side="head",
        )
        tail_window = _select_window(
            x_real,
            v_real,
            x_tail_missing,
            tail_correction_func,
            maxfev,
            side="tail",
        )
    else:
        head_window = tail_window = fit_window_size

    # head correction (i.e., left side of potential). The extrapolation is
    # anchored to the boundary data point's value and slope, so it joins the
    # real data with C1 continuity regardless of window or correction form.
    try:
        head_pot_correction = _anchored_predict(
            x_real,
            v_real,
            x_head_missing,
            head_correction_func,
            maxfev,
            side="head",
            w=head_window,
        )
    except RuntimeError:
        print(bad_fit_error_msg)
        raise RuntimeError(
            "Curve fitting failed for the bond head correction."
        ) from None

    # tail correction (i.e., right side of potential)
    try:
        tail_pot_correction = _anchored_predict(
            x_real,
            v_real,
            x_tail_missing,
            tail_correction_func,
            maxfev,
            side="tail",
            w=tail_window,
        )
    except RuntimeError:
        print(bad_fit_error_msg)
        raise RuntimeError(
            "Curve fitting failed for the bond tail correction."
        ) from None

    # Apply correction regions to original potential
    V[:head_start] = head_pot_correction
    V[tail_start + 1 :] = tail_pot_correction

    return V, head_start, tail_start + 1, real_indices


def pair_corrections(
    x: np.ndarray,
    V: np.ndarray,
    r_switch: float,
    smoothing_window: int,
    smoothing_order: int,
    fit_window_size: int | None,
    maxfev: int,
    head_correction_func: Callable,
):
    """The default correction method for bonded forces.

    .. note::

        No functional form is used to fit the tail of the potential
        as it is for the head. The tail of the potential is corrected
        so that it approaches zero smoothly between r_switch
        and r_cut.

    Parameters
    ----------
    x : np.ndarray
        The x values of the force.
    V : np.ndarray
        The y values of the force.
    r_switch : Union[int, float]
        The x-value to begin smooth approach towards zero.
    smoothing_window : int
        Window size to use in scipy.signal.savgol_fiter
        to smooth V before fitting.
    smoothing_order : int
        Polynomial order to use in scipy.signal.savgol_fiter
        to smooth V before fitting.
    head_correction_func : Callable
        Functional form to use in fitting the head (left) side
        of the potential.
    """
    V = np.copy(V)
    real_indices = _get_real_indices(V)
    v_real = np.copy(V[real_indices])
    x_real = np.copy(x[real_indices])
    head_start = real_indices[0]
    if all([smoothing_window, smoothing_order]):
        v_real = savgol_filter(
            x=v_real,
            window_length=smoothing_window,
            polyorder=smoothing_order,
            mode="mirror",
        )
    # head correction (short range repulsion).
    # When fit_window_size is None, sweep candidate windows and keep the one
    # that best extrapolates onto held-out real data near the gap. The
    # extrapolation is anchored to the boundary data point's value and slope, so
    # it joins the real data with C1 continuity regardless of window or form.
    x_head_missing = x[:head_start]
    if fit_window_size is None:
        fit_window_size = _select_window(
            x_real,
            v_real,
            x_head_missing,
            head_correction_func,
            maxfev,
            side="head",
        )
    try:
        head_pot_correction = _anchored_predict(
            x_real,
            v_real,
            x_head_missing,
            head_correction_func,
            maxfev,
            side="head",
            w=fit_window_size,
        )
    except RuntimeError:
        print(bad_fit_error_msg)
        raise RuntimeError(
            "Curve fitting failed for the pair head correction."
        ) from None

    # Apply correction regions to original potential
    V[:head_start] = head_pot_correction

    # Tail correction, long range approach to zero
    V_multiplier = np.ones_like(x)
    # If r_switch is not given, don't apply tail corrections
    if r_switch:
        r_cut = x[-1]
        idx_r_switch, r_switch = find_nearest(x, r_switch)
        # Entire V will be multiplied by this
        r_correct = x[idx_r_switch:]
        # Region of tail_multiplier that begins to decrease from 1
        V_multiplier[idx_r_switch:] = (
            (r_cut**2 - r_correct**2) ** 2
            * (r_cut**2 + 2 * r_correct**2 - 3 * r_switch**2)
            / (r_cut**2 - r_switch**2) ** 3
        )
    else:
        idx_r_switch = -1

    V *= V_multiplier

    return V, head_start, idx_r_switch, real_indices


def _get_real_indices(V: np.ndarray):
    """Find where infinity or NaN values exist in the potential."""
    real_idx = np.where(np.isfinite(V))[0]
    # Check for continuity of real_indices:
    if not np.all(np.ediff1d(real_idx) == 1):
        min_window = np.max(np.ediff1d(real_idx)) - 1
        if min_window > 5:
            raise RuntimeError(
                "The region of undefined values within the potential is too large. "
                "This could be the result of a sampling issue. Check the target distributions."
            )
        start = real_idx[0]
        end = real_idx[-1]
        # Correct nans, infs that are surrounded by 2 finite numbers
        for idx, v in enumerate(V[start:end]):
            if not np.isfinite(v):
                try:
                    avg = (
                        V[idx + start - min_window] + V[idx + start + min_window]
                    ) / 2
                    V[idx + start] = avg
                except IndexError:
                    pass
        # Trim off edge cases
        _real_idx = np.where(np.isfinite(V))[0]
        real_idx = max([list(g) for g in mit.consecutive_groups(_real_idx)], key=len)
    return real_idx


def _boundary_slope(u_win: np.ndarray, v_win: np.ndarray, degree: int):
    """Noise-averaged slope of the data at the seam (u = 0).

    Read from the derivative of a degree-`degree` polynomial fit to the whole
    window rather than a raw two-point difference, so the pinned slope is
    smoothed over the window instead of carrying a single point's noise.
    """
    return float(np.polyval(np.polyder(np.polyfit(u_win, v_win, degree)), 0.0))


def _anchored_predict(
    x_region: np.ndarray,
    v_region: np.ndarray,
    x_missing: np.ndarray,
    func: Callable,
    maxfev: int,
    side: str,
    w: int,
    slope_degree: int = 2,
):
    """Extrapolate into the gap with a boundary-anchored, C1-continuous curve.

    This is the single correction construction used everywhere: the sweep uses
    it to score windows, and the correction functions use it to fill the gap.
    The extrapolation is written as an offset from the real data point that
    borders the gap,

        V(u) = v_b + s_b * u + g(u),   u = x - x_b,

    where (x_b, v_b) is the boundary data point and s_b is the boundary slope
    from _boundary_slope. The curvature term g is the chosen functional form fit
    to the window, with its own value and slope at the boundary subtracted off,

        g(u) = func(u) - func(0) - func'(0) * u,

    so that g(0) = 0 and g'(0) = 0 for any form. The value and first derivative
    at the seam are therefore pinned to the real data by construction, giving C1
    continuity across the real/gap join every time, no matter the form or the
    window. The form no longer sets the whole extrapolation, only how it bends
    deeper into the gap. A linear form contributes no curvature and reduces to a
    straight tangent continuation.

    side='head' anchors on the first real point (extrapolate to smaller x);
    side='tail' anchors on the last real point (extrapolate to larger x). The
    form is fit in the raw x coordinate (where it stays well conditioned); the
    anchoring subtracts the form's own value and slope at the boundary, so g is
    the same regardless of any coordinate shift.
    """
    if side == "head":
        x_b, v_b = x_region[0], v_region[0]
        x_win, v_win = x_region[:w], v_region[:w]
    else:
        x_b, v_b = x_region[-1], v_region[-1]
        x_win, v_win = x_region[-w:], v_region[-w:]
    s_b = _boundary_slope(x_win - x_b, v_win, slope_degree)
    popt, _ = curve_fit(f=func, xdata=x_win, ydata=v_win, maxfev=maxfev)
    # Subtract the form's own value and slope at the boundary so only its
    # curvature survives; then re-pin value and slope to the data.
    du = 1e-6 * (abs(x_b) + 1.0)
    f_b = func(np.array([x_b]), *popt)[0]
    fp_b = (
        func(np.array([x_b + du]), *popt)[0] - func(np.array([x_b - du]), *popt)[0]
    ) / (2.0 * du)
    u_m = x_missing - x_b
    g = func(x_missing, *popt) - f_b - fp_b * u_m
    return v_b + s_b * u_m + g


def _walk_forward_score(
    x_region: np.ndarray,
    v_region: np.ndarray,
    func: Callable,
    maxfev: int,
    side: str,
    w: int,
    holdout: int,
):
    """Score a candidate fit window by held-out extrapolation error.

    The correction has to extrapolate past the edge of the real data into the
    region of missing (NaN/inf) values. We can't score that directly since
    there is no ground truth in the gap, so we simulate it: reserve a small
    holdout block of real points nearest the gap, build the anchored correction
    (see _anchored_predict) from the w points just inside that block, then
    extrapolate back onto the holdout and measure the RMSE against the real
    values. A window that predicts the held-out edge points well is the window
    whose correction extrapolates most trustworthily into the true gap, with no
    peak or well finding required. The score uses the same construction the gap
    is actually filled with, so selection and application stay consistent.

    side='head' treats the start of the arrays as the gap edge (extrapolate to
    smaller x); side='tail' treats the end as the gap edge.
    """
    if side == "head":
        x_hold, v_hold = x_region[:holdout], v_region[:holdout]
        x_train, v_train = x_region[holdout:], v_region[holdout:]
    else:
        x_hold, v_hold = x_region[-holdout:], v_region[-holdout:]
        x_train, v_train = x_region[:-holdout], v_region[:-holdout]
    v_pred = _anchored_predict(x_train, v_train, x_hold, func, maxfev, side, w)
    return float(np.sqrt(np.mean((v_pred - v_hold) ** 2)))


def _extrapolation_instability(pred: np.ndarray, neighbors: list):
    """Mean relative L2 distance between a gap extrapolation and its neighbors.

    A window whose gap extrapolation barely moves when the window size is
    nudged up or down sits on a stable plateau and is trustworthy. A window
    whose extrapolation swings wildly against its neighbors is fragile, even if
    its own held-out RMSE happens to be low. Returns 0.0 when there are no
    neighbors to compare against.
    """
    if not neighbors:
        return 0.0
    diffs = []
    for nb in neighbors:
        scale = 0.5 * (np.linalg.norm(pred) + np.linalg.norm(nb)) + 1e-12
        diffs.append(np.linalg.norm(pred - nb) / scale)
    return float(np.mean(diffs))


def _select_window(
    x_region: np.ndarray,
    v_region: np.ndarray,
    x_missing: np.ndarray,
    func: Callable,
    maxfev: int,
    side: str,
    window_min: int = 4,
    window_max: int = 25,
    holdout: int = 3,
    rmse_tol: float = 0.25,
    return_scores: bool = False,
):
    """Sweep candidate fit windows and return the best window size.

    Replaces the single user-supplied fit_window_size with a scan over window
    sizes from window_min to window_max. Selection happens in two stages.
    First each window is scored by _walk_forward_score (held-out extrapolation
    RMSE). Rather than taking the raw minimum, which can be a noise-driven spike,
    every window whose RMSE is within rmse_tol of the best is treated as a
    statistical tie. The tie is then broken on stability: among those windows we
    keep the one whose actual gap extrapolation is least sensitive to a change in
    window size, as measured by _extrapolation_instability. This favours windows
    sitting on a smooth plateau over lucky spikes. Candidate windows that fail to
    fit are skipped rather than aborting the sweep.

    Set return_scores=True to also get a {window_size: {rmse, instability}} dict,
    which is handy for diagnostics and plotting.
    """
    n = len(v_region)
    w_hi = min(window_max, n - holdout)
    if w_hi < window_min:
        # Too little real data to sweep, so fall back to as many points as we
        # can spare while still leaving the holdout block.
        w = max(2, n - holdout)
        info = {w: {"rmse": float("nan"), "instability": float("nan")}}
        return (w, info) if return_scores else w

    rmse = {}
    preds = {}
    for w in range(window_min, w_hi + 1):
        try:
            score = _walk_forward_score(
                x_region, v_region, func, maxfev, side, w, holdout
            )
            pred = _anchored_predict(
                x_region, v_region, x_missing, func, maxfev, side, w
            )
        except (RuntimeError, TypeError, ValueError):
            # A window this func can't fit is simply not a candidate.
            continue
        rmse[w] = score
        preds[w] = pred
    if not rmse:
        raise RuntimeError(
            "Window sweep failed to fit any candidate window.\n" + bad_fit_error_msg
        )

    instability = {}
    for w, pred in preds.items():
        neighbors = [preds[w + d] for d in (-1, 1) if w + d in preds]
        instability[w] = _extrapolation_instability(pred, neighbors)

    # Windows statistically tied with the best held-out RMSE, then break the tie
    # on extrapolation stability.
    best_rmse = min(rmse.values())
    tied = [w for w, r in rmse.items() if r <= best_rmse * (1.0 + rmse_tol)]
    best_w = min(tied, key=lambda w: instability[w])

    if return_scores:
        info = {w: {"rmse": rmse[w], "instability": instability[w]} for w in rmse}
        return best_w, info
    return best_w

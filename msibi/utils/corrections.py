import more_itertools as mit
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from msibi.utils.general import find_nearest


def harmonic(x, x0, k):
    "V(x) = 0.5(x - x0)^2"
    return 0.5 * k * (x - x0) ** 2


def exponential(x, A, B):
    "V(x) = A*exp(-Bx)"
    return A * np.exp(-B * x)


def linear(x, m, b):
    "V(x) = mx + b"
    return x * m + b


def bonded_corrections(
    x,
    V,
    smoothing_window,
    smoothing_order,
    fit_window_size,
    head_correction_func,
    tail_correction_func,
):
    V = np.copy(V)
    real_indices = _get_real_indices(V)
    v_real = np.copy(V[real_indices])
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

    # head correction (i.e., left side of potential)
    # Get fit parameters for where we actually have data
    # Need to shift x-values. Function must increase as x becomes smaller
    x_head_pivot = x_real[fit_window_size - 1]
    x_head_fit = _shift_x(x_real[:fit_window_size], origin=x_head_pivot)
    popt_head, pcov_head = curve_fit(
        f=head_correction_func,
        xdata=x_head_fit,
        ydata=v_real[:fit_window_size],
    )
    x_head_missing = _shift_x(x[:head_start], origin=x_head_pivot)
    # Apply these parameters to the x-range where we are missing data
    head_pot_correction = head_correction_func(x_head_missing, *popt_head)

    # tail correction (i.e., right side of potential)
    popt_tail, pcov_tail = curve_fit(
        f=tail_correction_func,
        xdata=x_real[-fit_window_size:],
        ydata=v_real[-fit_window_size:],
    )
    tail_pot_correction = tail_correction_func(x[tail_start + 1 :], *popt_tail)

    # Apply correction regions to original potential
    V[:head_start] = head_pot_correction
    V[tail_start + 1 :] = tail_pot_correction

    return V, head_start, tail_start + 1, real_indices


def pair_corrections(
    x,
    V,
    r_switch,
    smoothing_window,
    smoothing_order,
    fit_window_size,
    head_correction_func,
):
    V = np.copy(V)
    real_indices = _get_real_indices(V)
    v_real = np.copy(V[real_indices])
    x_real = np.copy(x[real_indices])
    head_start = real_indices[0]
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
    # head correction (short range repulsion)
    # Get fit parameters for where we actually have data
    popt_head, pcov_head = curve_fit(
        f=head_correction_func,
        xdata=x_real[: fit_window_size + 1],
        ydata=v_real[: fit_window_size + 1],
    )
    head_pot_correction = head_correction_func(x[:head_start], *popt_head)

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
        # Apply correction regions to original potential
        V[:head_start] = head_pot_correction
    else:
        idx_r_switch = -1

    V *= V_multiplier

    return V, head_start, idx_r_switch, real_indices


def _get_real_indices(V):
    real_idx = np.where(np.isfinite(V))[0]
    # Check for continuity of real_indices:
    if not np.all(np.ediff1d(real_idx) == 1):
        start = real_idx[0]
        end = real_idx[-1]
        # Correct nans, infs that are surrounded by 2 finite numbers
        for idx, v in enumerate(V[start:end]):
            if not np.isfinite(v):
                try:
                    avg = (V[idx + start - 1] + V[idx + start + 1]) / 2
                    V[idx + start] = avg
                except IndexError:
                    pass
        # Trim off edge cases
        _real_idx = np.where(np.isfinite(V))[0]
        real_idx = max([list(g) for g in mit.consecutive_groups(_real_idx)], key=len)
    return real_idx


def _shift_x(x, origin):
    return x - origin

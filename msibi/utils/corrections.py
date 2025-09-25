from typing import Callable, Union

import more_itertools as mit
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from msibi.utils.general import find_nearest


def harmonic(x: np.ndarray, x0: Union[float, int], k: Union[float, int]):
    """Used as the default correction form for bonded forces.

        :math:`V(x) = 0.5*k(x - x0)^2`

    .. note::

        This is used by default for msibi.force.Bond, msibi.force.Angle
        and msibi.force.Dihedral.

    """
    return 0.5 * k * (x - x0) ** 2


def exponential(x: np.ndarray, A: Union[float, int], B: Union[float, int]):
    """Used as the default head correction for non-bonded pair potentials.

    :math:`V(x) = A*exp(-Bx)`

    """
    return A * np.exp(-B * x)


def linear(x: np.ndarray, m: Union[float, int], b: Union[float, int]):
    """Functional form that can be used for head or tail corrections.

    :math:`V(x) = mx + b`


    """
    return x * m + b


def bonded_corrections(
    x: np.ndarray,
    V: np.ndarray,
    smoothing_window: int,
    smoothing_order: int,
    fit_window_size: int,
    maxfev: int,
    head_correction_func: Callable,
    tail_correction_func: Callable,
):
    """The default correction method for bonded forces.

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

    # head correction (i.e., left side of potential)
    # Get fit parameters for where we actually have data
    # Need to shift x-values. Function must increase as x becomes smaller
    x_head_pivot = x_real[fit_window_size - 1]
    x_head_fit = _shift_x(x_real[:fit_window_size], origin=x_head_pivot)
    popt_head, pcov_head = curve_fit(
        f=head_correction_func,
        xdata=x_head_fit,
        ydata=v_real[:fit_window_size],
        maxfev=maxfev,
    )
    x_head_missing = _shift_x(x[:head_start], origin=x_head_pivot)
    # Apply these parameters to the x-range where we are missing data
    head_pot_correction = head_correction_func(x_head_missing, *popt_head)

    # tail correction (i.e., right side of potential)
    popt_tail, pcov_tail = curve_fit(
        f=tail_correction_func,
        xdata=x_real[-fit_window_size:],
        ydata=v_real[-fit_window_size:],
        maxfev=maxfev,
    )
    tail_pot_correction = tail_correction_func(x[tail_start + 1 :], *popt_tail)

    # Apply correction regions to original potential
    V[:head_start] = head_pot_correction
    V[tail_start + 1 :] = tail_pot_correction

    return V, head_start, tail_start + 1, real_indices


def pair_corrections(
    x: np.ndarray,
    V: np.ndarray,
    r_switch: Union[float, int],
    smoothing_window: int,
    smoothing_order: int,
    fit_window_size: int,
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
    # head correction (short range repulsion)
    # Get fit parameters for where we actually have data
    popt_head, pcov_head = curve_fit(
        f=head_correction_func,
        xdata=x_real[: fit_window_size + 1],
        ydata=v_real[: fit_window_size + 1],
        maxfev=maxfev,
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


def _shift_x(x: np.ndarray, origin: Union[float, int]):
    """Shift x-values in order to generate increasing f(x) as x decreases.

    This is used for bonded_corrections().
    """
    return x - origin

import warnings

import more_itertools as mit
import numpy as np
from scipy.optimize import curve_fit

from msibi.utils.general import find_nearest


def quadratic_spring(x, x0, k4, k3, k2):
    """Creates a quadratic spring-like potential with the following form

        V(x) = k4(x-x0)^4 + k3(x-x0)^3 + k2(x-x0)^2

    Used in creating table potentials for bonded potentials.

    """
    V_x = k4 * ((x - x0) ** 4) + k3 * ((x - x0) ** 3) + k2 * ((x - x0) ** 2)
    return V_x


def mie(r, epsilon, sigma, m, n):
    """The Mie potential functional form"""
    prefactor = (m / (m - n)) * (m / n) ** (n / (m - n))
    V_r = prefactor * epsilon * ((sigma / r) ** m - (sigma / r) ** n)
    return V_r


def lennard_jones(r, epsilon, sigma):
    return mie(r=r, epsilon=epsilon, sigma=sigma, m=12, n=6)


def pair_correction(r, V, form, r_switch=2.5):
    if form == "linear":
        head_correction_function = linear_head_correction
        tail_correction_function = pair_tail_correction
    elif form == "exponential":
        head_correction_function = exponential_head_correction
        tail_correction_function = pair_tail_correction
    else:
        raise ValueError(f'Unsupported head correction form: "{form}"')

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
        real_idx = max(
            [list(g) for g in mit.consecutive_groups(_real_idx)], key=len
        )

    head_cutoff = real_idx[0] - 1
    tail_cutoff = real_idx[-1] + 1

    head_correction_V = head_correction_function(r=r, V=V, cutoff=head_cutoff)
    # Potential with both head correction and tial correciton applied
    tail_correction_V = tail_correction_function(
        r=r, V=head_correction_V, r_switch=r_switch
    )
    return tail_correction_V, real_idx, head_cutoff, tail_cutoff


def bond_correction(r, V, form):
    """Handles corrections for both the head and tail of
    bond scretching and angle potentials.
    """
    import more_itertools as mit

    if form == "linear":
        head_correction_function = linear_head_correction
        tail_correction_function = linear_tail_correction
    elif form == "linear_optimized":
        head_correction_function = linear_head_correction_optimized
        tail_correction_function = linear_tail_correction_optimized
    elif form == "exponential":
        head_correction_function = exponential_head_correction
        tail_correction_function = exponential_tail_correction
    else:
        raise ValueError(f'Unsupported head correction form: "{form}"')

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
        real_idx = max(
            [list(g) for g in mit.consecutive_groups(_real_idx)], key=len
        )

    head_cutoff = real_idx[0] - 1
    tail_cutoff = real_idx[-1] + 1
    # Potential with the head correction applied
    head_correction_V = head_correction_function(r=r, V=V, cutoff=head_cutoff)
    # Potential with both head correction and tial correciton applied
    tail_correction_V = tail_correction_function(
        r=r, V=head_correction_V, cutoff=tail_cutoff
    )
    return tail_correction_V, real_idx, head_cutoff, tail_cutoff


def pair_tail_correction(r, V, r_switch):
    """Apply a tail correction to a potential making it go to zero smoothly.

    Parameters
    ----------
    r : np.ndarray, shape=(n_points,), dtype=float
        The radius values at which the potential is given.
    V : np.ndarray, shape=r.shape, dtype=float
        The potential values at each radius value.
    r_switch : float, optional, default=pot_r[-1] - 5 * dr
        The radius after which a tail correction is applied.

    References
    ----------
    .. [1] https://codeblue.umich.edu/hoomd-blue/doc/classhoomd__script_1_1pair_1_1pair.html

    """
    r_cut = r[-1]
    idx_r_switch, r_switch = find_nearest(r, r_switch)

    S_r = np.ones_like(r)
    r = r[idx_r_switch:]
    S_r[idx_r_switch:] = (
            (r_cut ** 2 - r ** 2) ** 2
            * (r_cut ** 2 + 2 * r ** 2 - 3 * r_switch ** 2)
            / (r_cut ** 2 - r_switch ** 2) ** 3
    )
    return V * S_r


def pair_head_correction(r, V, previous_V=None, form="linear"):
    """Apply head correction to V making it go to a finite value at V(0).

    Parameters
    ----------
    r : np.ndarray, shape=(n_points,), dtype=float
        The radius values at which the potential is given.
    V : np.ndarray, shape=r.shape, dtype=float
        The potential values at each radius value.
    previous_V : np.ndarray, shape=r.shape, dtype=float
        The potential from the previous iteration.
    form : str, optional, default='linear'
        The form of the smoothing function used.

    """
    if form == "linear":
        correction_function = linear_head_correction
    elif form == "exponential":
        correction_function = exponential_head_correction
    else:
        raise ValueError(f'Unsupported head correction form: "{form}"')

    for i, pot_value in enumerate(V[::-1]):
        # Apply correction function because either of the following is true:
        #   * both current and target RDFs are 0 --> nan values in potential.
        #   * current rdf > 0, target rdf = 0 --> +inf values in potential.
        if np.isnan(pot_value) or np.isposinf(pot_value):
            last_real = V.shape[0] - i - 1
            if last_real > len(V) - 2:
                raise RuntimeError(
                    "Undefined values in tail of potential."
                    "This probably means you need better "
                    "sampling at this state point."
                )
            return correction_function(r, V, last_real)
        # Retain old potential at small r because:
        #   * current rdf = 0, target rdf > 0 --> -inf values in potential.
        elif np.isneginf(pot_value):
            last_neginf = V.shape[0] - i - 1
            for i, pot_value in enumerate(V[: last_neginf + 1]):
                V[i] = previous_V[i]
            return V
    else:
        warnings.warn(
            "No inf/nan values in your potential--this is unusual!"
            "No head correction applied"
        )
        return V


def linear_tail_correction(r, V, cutoff, window=6):
    """Use a linear function to smoothly force V to a finite value at V(cut).

    Parameters
    ----------
    r : np.ndarray
        Separation values
    V : np.ndarray
        Potential at each of the separation values
    cutoff : int
        The last real value of V when iterating backwards
    window : int
        Number of data points backward from cutoff to use in slope calculation

    """
    slope = (
                    V[cutoff - 1] - V[cutoff - window]
            ) / (r[cutoff - 1] - r[cutoff - window])
    if slope < 0:
        slope = -slope
    V[cutoff:] = slope * (r[cutoff:] - r[cutoff - 1]) + V[cutoff - 1]
    return V


def linear_tail_correction_optimized(r, V, cutoff, window=6):
    """Use a linear function to smoothly force V to a finite value at V(cut).

    This function uses scipy.optimize.curve_fit to find the slope and intercept
    of the linear function that is used to correct the tail of the potential.

    Parameters
    ----------
    r : np.ndarray
        Separation values
    V : np.ndarray
        Potential at each of the separation values
    cutoff : int
        The last real value of V when iterating backwards
    window : int
        Number of data points backward from cutoff to use in slope calculation

    """

    def linear(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(linear, r[cutoff - window:cutoff], V[cutoff - window:cutoff])
    V[cutoff:] = linear(r[cutoff:], *popt)
    return V


def linear_head_correction(r, V, cutoff, window=6):
    """Use a linear function to smoothly force V to a finite value at V(0).
    Parameters
    ----------
    r : np.ndarray
        Separation values
    V : np.ndarray
        Potential at each of the separation values
    cutoff : int
        The first real value of V when iterating forwards
    window : int
        Number of data points forward from cutoff to use in slope calculation

    """
    slope = (
                    V[cutoff + 1] - V[cutoff + window]
            ) / (r[cutoff + 1] - r[cutoff + window])
    if slope > 0:
        slope = -slope
    V[: cutoff + 1] = slope * (r[: cutoff + 1] - r[cutoff + 1]) + V[cutoff + 1]
    return V


def linear_head_correction_optimized(r, V, cutoff, window=6):
    """Use a linear function to smoothly force V to a finite value at V(0).

    This function uses scipy.optimize.curve_fit to find the slope and intercept
    of the linear function that is used to correct the head of the potential.

    Parameters
    ----------
    r : np.ndarray
        Separation values
    V : np.ndarray
        Potential at each of the separation values
    cutoff : int
        The first real value of V when iterating forwards
    window : int
        Number of data points forward from cutoff to use in slope calculation
    """

    def linear(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(linear, r[cutoff + 1:cutoff + window], V[cutoff + 1:cutoff + window])
    V[:cutoff + 1] = linear(r[:cutoff + 1], *popt)
    return V


def exponential_tail_correction(r, V, cutoff):
    """Use an exponential function to smoothly force V to a finite value at V(cut)

    Parameters
    ----------
    r : np.ndarray
        Separation values
    V : np.ndarray
        Potential at each of the separation values
    cutoff : int
        The last non-real value of V when iterating backwards

    This function fits the small part of the potential to the form:
    V(r) = A*exp(Br)

    """
    raise RuntimeError("Exponential tail corrections are not implemented."
                       "Use the linear correction form when optimizing bonds and angles."
                       )
    dr = r[cutoff - 1] - r[cutoff - 2]
    B = np.log(V[cutoff - 1] / V[cutoff - 2]) / dr
    A = V[cutoff - 1] * np.exp(B * r[cutoff - 1])
    V[cutoff:] = A * np.exp(B * r[cutoff:])
    return V


def exponential_head_correction(r, V, cutoff):
    """Use an exponential function to smoothly force V to a finite value at V(0)

    Parameters
    ----------
    r : np.ndarray
        Separation values
    V : np.ndarray
        Potential at each of the separation values
    cutoff : int
        The last non real value of V when iterating forwards

    This function fits the small part of the potential to the form:
    V(r) = A*exp(-Br)

    """
    dr = r[cutoff + 2] - r[cutoff + 1]
    B = np.log(V[cutoff + 1] / V[cutoff + 2]) / dr
    A = V[cutoff + 1] * np.exp(B * r[cutoff + 1])
    V[:cutoff + 1] = A * np.exp(-B * r[:cutoff + 1])
    return V


def alpha_array(alpha0, pot_r, form="linear"):
    """Generate an array of alpha values used for scaling in the IBI step. """
    return alpha0 * (1.0 - pot_r / pot_r[-1])

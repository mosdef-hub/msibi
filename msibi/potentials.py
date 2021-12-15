import numpy as np

from msibi.utils.general import find_nearest

__all__ = ["mie", "morse"]


def mie(r, eps, sig, m=12, n=6):
    """Mie pair potential.  """
    prefactor = (m / (m - n)) * (m / n) ** (n / (m - n))
    return prefactor * eps * ((sig / r) ** m - (sig / r) ** n)


def morse(r, D, alpha, r0):
    """Morse pair potential. """
    return D * (np.exp(-2 * alpha * (r - r0)) - 2 * np.exp(-alpha * (r - r0)))


def tail_correction(r, V, r_switch):
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


def head_correction(r, V, previous_V, form="linear"):
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
        raise ValueError('Unsupported head correction form: "{0}"'.format(form))

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
        # TODO: Raise error?
        #       This means that all potential values are well behaved.
        pass


def linear_head_correction(r, V, cutoff):
    """Use a linear function to smoothly force V to a finite value at V(0). """
    slope = (V[cutoff + 1] - V[cutoff + 2]) / (r[cutoff + 1] - r[cutoff + 2])
    if slope > 0:
        slope = -slope
    V[: cutoff + 1] = slope * (r[: cutoff + 1] - r[cutoff + 1]) + V[cutoff + 1]
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
        The last real value of V when iterating backwards

    This function fits the small part of the potential to the form:
    V(r) = A*exp(-Br)
    """
    dr = r[cutoff + 2] - r[cutoff + 1]
    B = np.log(V[cutoff + 1] / V[cutoff + 2]) / dr
    A = V[cutoff + 1] * np.exp(B * r[cutoff + 1])
    V[: cutoff + 1] = A * np.exp(-B * r[: cutoff + 1])
    return V


def alpha_array(alpha0, pot_r, form="linear"):
    """Generate an array of alpha values used for scaling in the IBI step. """
    if form == "linear":
        return alpha0 * (1.0 - pot_r / pot_r[-1])
    else:
        raise ValueError("Unsupported alpha form")

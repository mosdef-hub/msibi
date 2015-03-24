import numpy as np

__all__ = ['mie', 'morse']


def mie(r, eps, sig, m=12, n=6):
    """Mie pair potential. """
    return 4 * eps * ((sig / r) ** m - (sig / r) ** n)


def morse(r, D, alpha, r0):
    """Morse pair potential. """
    return D * (np.exp(-2 * alpha * (r - r0)) -
                2 * np.exp(-alpha * (r - r0)))


def tail_correction(r, dr, V, r_switch):
    """ """
    r_cut = r[-1]

    idx_r_switch, r_switch = find_nearest(r, r_switch)

    S_r = np.ones_like(r)
    # TODO: See HOOMD XPLOR smooth function reference.
    S_r[idx_r_switch:] = ((r_cut ** 2 - r[idx_r_switch:] ** 2) ** 2 *
                      (r_cut ** 2 + 2 * r[idx_r_switch:] ** 2 - 3 * r_switch ** 2) /
                      (r_cut ** 2 - r_switch ** 2) ** 3)

    V *= S_r
    return r, V


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'. """
    idx = np.abs(array - target).argmin()
    return idx, array[idx]

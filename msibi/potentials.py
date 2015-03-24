import numpy as np

__all__ = ['mie', 'morse']


def mie(r, eps, sig, m=12, n=6):
    """Mie pair potential. """
    return 4 * eps * ((sig / r) ** m - (sig / r) ** n)


def morse(r, D, alpha, r0):
    """Morse pair potential. """
    return D * (np.exp(-2 * alpha * (r - r0)) -
                2 * np.exp(-alpha * (r - r0)))


def tail_correction(r, dr, V, r_on=None):
    """ """
    if not r_on:
        r_on = r.max() - 5 * dr

    r = np.append(r, [r[-1] + dr])
    r_cut = r[-1]

    V = np.append(V, [0])

    idx_r_on, r_on = find_nearest(r, r_on)

    S_r = np.ones_like(r)
    # TODO: See HOOMD XPLOR smooth function reference.
    S_r[idx_r_on:] = ((r_cut ** 2 - r[idx_r_on:] ** 2) ** 2 *
                      (r_cut ** 2 + 2 * r[idx_r_on:] ** 2 - 3 * r_on ** 2) /
                      (r_cut ** 2 - r_on ** 2) ** 3)

    V *= S_r
    return r, V


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'. """
    idx = np.abs(array - target).argmin()
    return idx, array[idx]

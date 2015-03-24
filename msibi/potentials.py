import numpy as np

__all__ = ['mie', 'morse']


def mie(r, eps, sig, m=12, n=6):
    """Mie pair potential. """
    return 4 * eps * ((sig / r) ** m - (sig / r) ** n)


def morse(r, D, alpha, r0):
    """Morse pair potential. """
    return D * (np.exp(-2 * alpha * (r - r0)) -
                2 * np.exp(-alpha * (r - r0)))

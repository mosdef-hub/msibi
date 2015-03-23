import numpy as np

from msibi.optimize import R


def mie(eps, sig, m=12, n=6):
    """Mie pair potential. """
    return 4 * eps * ((sig / R) ** m - (sig / R) ** n)


def morse(D, alpha, r0):
    """Morse pair potential. """
    return D * (np.exp(-2 * alpha * (R - r0)) -
                2 * np.exp(-alpha * (R - r0)))

import numpy as np

from msibi.msibi import R


def mie(eps, sig, m=12, n=6):
    """Mie pair potential. """
    return 4 * eps * [(sig / R) ** m - (sig / R) ** n]


def morse(D, alpha, r0):
    """Morse pair potential. """
    return D * [np.exp(-2 * alpha * (R - r0)) -
                2 * np.exp(-alpha * (R - r0))]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    y = mie(1.0, 1.0)
    plt.plot(R, y)
    plt.show()
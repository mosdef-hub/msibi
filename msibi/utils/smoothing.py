from math import factorial

import numpy as np


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """

    Parameters
    ----------
    y:
    window_size:
    order:
    deriv:
    rate:

    Returns
    -------

    """
    if not (isinstance(window_size, int) and isinstance(order, int)):
        raise ValueError("window_size and order must be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat(
        [
            [k ** i for i in order_range]
            for k in range(-half_window, half_window + 1)
        ]
    )
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")

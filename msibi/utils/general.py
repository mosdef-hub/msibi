import numpy as np


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'."""
    idx = np.abs(array - target).argmin()
    return idx, array[idx]

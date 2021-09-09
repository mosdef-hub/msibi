import glob
import os
import shutil

import numpy as np
from pkg_resources import resource_filename
import gc
import msibi


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'."""
    idx = np.abs(array - target).argmin()
    return idx, array[idx]


def _count_backups(filename):
    """Count the number of backups of a file in a directory."""
    head, tail = os.path.split(filename)
    backup_files = f"_.*.{tail}"
    return len(glob.glob(os.path.join(head, backup_files)))


def _backup_name(filename, n_backups):
    """Return backup filename based on the number of existing backups.

    Parameters
    ----------
    filename : str
        Full path to file to make backup of.
    n_backups : int
        Number of existing backups.
    """
    head, tail = os.path.split(filename)
    new_backup = f"_.{n_backups:d}.{tail}"
    return os.path.join(head, new_backup)


def backup_file(filename):
    """Backup a file based on the number of backups in the file's directory.

    Parameters
    ----------
    filename : str
        Full path to file to make backup of.
    """
    n_backups = _count_backups(filename)
    new_backup = _backup_name(filename, n_backups)
    shutil.copy(filename, new_backup)

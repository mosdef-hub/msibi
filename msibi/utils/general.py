import glob
import os
import shutil

import numpy as np


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'. """
    idx = np.abs(array - target).argmin()
    return idx, array[idx]


def _count_backups(filename):
    """Count the number of backups of a file in a directory
    
    Args:
        filename (str): Full path to file to check for backups
    """
    head, tail = os.path.split(filename)
    backup_files = ''.join(['_.*.', tail])
    return len(glob.glob(os.path.join(head, backup_files)))


def _backup_name(filename, n_backups):
    """Return backup filename based on the filename and number of existing backups

    Args:
        filename (str): Full path to file to make backup of
        n_backups (int): number of existing backups; gets used in naming backup
    """
    head, tail = os.path.split(filename)
    new_backup = ''.join(['_.{0:d}.'.format(n_backups), tail])
    return os.path.join(head, new_backup)


def backup_file(filename):
    """Backup a file based on the number of backups in the file's directory

    Args:
        filename (str): Full path to file to make a backup of
    """
    n_backups = _count_backups(filename)
    new_backup = _backup_name(filename, n_backups)
    shutil.copy(filename, new_backup)

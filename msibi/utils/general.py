import glob
import os
import shutil

import numpy as np
from pkg_resources import resource_filename
import gc
import msibi

def get_fn(name):
    """Get the full path to one of the reference files shipped for testing.

    This function is taken straight from MDTraj
    (see https://github.com/mdtraj/mdtraj).
    In the source distribution, these files are in ``msibi/utils/reference``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the reference directory).

    Examples
    ________
    >>> import mdtraj as md
    >>> t = md.load(get_fun('final.hoomdxml'))
    """

    fn = resource_filename("msibi", os.path.join("utils", "reference", name))

    if not os.path.exists(fn):
        raise ValueError(
            "Sorry! %s does not exist. If you just "
            "added it, you'll have to re install" % fn
        )

    return fn


def find_nearest(array, target):
    """Find array component whose numeric value is closest to 'target'. """
    idx = np.abs(array - target).argmin()
    return idx, array[idx]


def _count_backups(filename):
    """Count the number of backups of a file in a directory. """
    head, tail = os.path.split(filename)
    backup_files = "".join(["_.*.", tail])
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
    new_backup = "".join(["_.{0:d}.".format(n_backups), tail])
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

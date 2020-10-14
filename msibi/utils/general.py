##############################################################################
# MSIBI: A package for optimizing coarse-grained force fields using multistate
#   iterative Boltzmann inversion.
# Copyright (c) 2017 Vanderbilt University and the Authors
#
# Authors: Christoph Klein, Timothy C. Moore
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files, to deal
# in MSIBI without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of MSIBI, and to permit persons to whom MSIBI is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of MSIBI.
#
# MSIBI IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH MSIBI OR THE USE OR OTHER DEALINGS ALONG WITH
# MSIBI.
#
# You should have received a copy of the MIT license.
# If not, see <https://opensource.org/licenses/MIT/>.
##############################################################################

import glob
import os
from pkg_resources import resource_filename
import shutil

import numpy as np


def get_fn(name):
    """Get the full path to one of the reference files shipped for testing.

    This function is taken straight from MDTraj (see https://github.com/mdtraj/mdtraj).
    In the source distribution, these files are in ``msibi/utils/reference``,
    but on istallation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file ot load (with respecto to the reference/ directory).

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

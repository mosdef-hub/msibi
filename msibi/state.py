##############################################################################
# MSIBI: A package for optimizing coarse-grained force fields using multistate
#   iterative Boltzmann inversion.
# Copyright (c) 2017 Vanderbilt University and the Authors
#
# Authors: Christoph Klein, Timothy C. Moore
# Contributors: Davy Yue
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

import os

import mdtraj as md

HOOMD1_HEADER = """
from hoomd_script import *

system = init.read_xml(filename="{0}", wrap_coordinates=True)

T_final = {1:.1f}

pot_width = {2:d}
table = pair.table(width=pot_width)
"""

HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.deprecated.init import read_xml
from hoomd.init import read_snapshot

hoomd.context.initialize("")
try:
    system = read_xml(filename="{0}", wrap_coordinates=True)
except RuntimeError:
    with gsd.hoomd.open("{0}") as t:
        snap = t[-1]
    system = read_snapshot(snap)
T_final = {1:.1f}

pot_width = {2:d}
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=pot_width, nlist=nl)

"""

HOOMD_TABLE_ENTRY = """
table.set_from_file('{type1}', '{type2}', filename='{potential_file}')
"""


class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    kT : float
        Unitless heat energy (product of Boltzmann's constant and temperature).
    state_dir : path
        Path to state directory (default '')
    traj_file : path or md.Trajectory
        The dcd or gsd trajectory associated with this state
        (default 'query.dcd')
    top_file : path
        hoomdxml containing topology information (needed for dcd)
        (default None)
    name : str
        State name. If no name is given, state will be named 'state-{kT:.3f}'
        (default None)
    backup_trajectory : bool
        True if each query trajectory is backed up (default False)

    """

    def __init__(
        self,
        kT,
        state_dir="",
        traj_file="query.dcd",
        top_file=None,
        name=None,
        backup_trajectory=False,
    ):

        self.kT = kT
        self.state_dir = state_dir

        self.traj_path = os.path.join(state_dir, traj_file)

        if top_file:
            self.top_path = os.path.join(state_dir, top_file)

        self.traj = None
        if not name:
            name = "state-{0:.3f}".format(self.kT)
        self.name = name

        self.backup_trajectory = backup_trajectory

    def reload_query_trajectory(self):
        """Reload the query trajectory. """
        if self.top_path:
            self.traj = md.load(self.traj_path, top=self.top_path)
        else:
            self.traj = md.load(self.traj_path)

    def save_runscript(
        self,
        table_potentials,
        table_width,
        engine="hoomd",
        runscript="hoomd_run_template.py",
    ):
        """Save the input script for the MD engine. """

        header = list()

        if self.HOOMD_VERSION == 1:
            HOOMD_HEADER = HOOMD1_HEADER
        elif self.HOOMD_VERSION == 2:
            HOOMD_HEADER = HOOMD2_HEADER

        header.append(HOOMD_HEADER.format(self.top_path, self.kT, table_width))
        for type1, type2, potential_file in table_potentials:
            header.append(HOOMD_TABLE_ENTRY.format(**locals()))
        header = "".join(header)
        with open(os.path.join(self.state_dir, runscript)) as fh:
            body = "".join(fh.readlines())

        runscript_file = os.path.join(self.state_dir, "run.py")
        with open(runscript_file, "w") as fh:
            fh.write(header)
            fh.write(body)

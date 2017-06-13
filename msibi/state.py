import os

import mdtraj as md

# **** 2017_05_31 Notes ******
############### update section start

HOOMD1_HEADER = """
from hoomd_script import *

system = read_xml(filename="{0}", wrap_coordinates=True)
T_final = {1:.1f}

pot_width = {2:d}
table = pair.table(width=pot_width)
"""


HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.deprecated.init import read_xml

hoomd.context.initialize("")
system = read_xml(filename="{0}", wrap_coordinates=True)
T_final = {1:.1f}

pot_width = {2:d}
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=pot_width, nlist=nl)

"""
HOOMD_TABLE_ENTRY = """
table.set_from_file('{type1}', '{type2}', filename='{potential_file}')
"""

##################### update section end

class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    k : float
        Boltzmann's  constant in specified units.
    T : float
        Temperature in kelvin.
    traj : md.Trajectory
        The trajectory associated with this state.
    backup_trajectory : bool
        True if each query trajectory is backed up (default=False)

    """
    def __init__(self, kT, state_dir='', traj_file=None, top_file=None,
                 name=None, backup_trajectory=False):
        self.kT = kT
        self.state_dir = state_dir

        if not traj_file:
            self.traj_path = os.path.join(state_dir, 'query.dcd')
        if top_file:
            self.top_path = os.path.join(state_dir, top_file)

        self.traj = None  # Will be set after first iteration.
        if not name:
            name = 'state-{0:.3f}'.format(self.kT)
        self.name = name

        self.backup_trajectory = backup_trajectory  # save trajectories?

    def reload_query_trajectory(self):
        """Reload the query trajectory. """
        if self.top_path:
            self.traj = md.load(self.traj_path, top=self.top_path)
        else:
            self.traj = md.load(self.traj_path)

            #differentiate hoomd verion headers here - maybe new class members
    def save_runscript(self, table_potentials, table_width, engine='hoomd',
                       runscript='hoomd_run_template.py'):
        """Save the input script for the MD engine. """

        # TODO: Factor out for separate engines.
        header = list()

        if self.HOOMD_VERSION == 1:
            HOOMD_HEADER = HOOMD1_HEADER
        elif self.HOOMD_VERSION == 2:
            HOOMD_HEADER = HOOMD2_HEADER

        header.append(HOOMD_HEADER.format('start.hoomdxml', self.kT, table_width))
        for type1, type2, potential_file in table_potentials:
            header.append(HOOMD_TABLE_ENTRY.format(**locals()))
        header = ''.join(header)
        with open(os.path.join(self.state_dir, runscript)) as fh:
            body = ''.join(fh.readlines())

        runscript_file = os.path.join(self.state_dir, 'run.py')
        with open(runscript_file, 'w') as fh:
            fh.write(header)
            fh.write(body)

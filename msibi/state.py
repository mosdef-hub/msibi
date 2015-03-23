import os

import mdtraj as md

from msibi.optimize import R

HOOMD_HEADER = """from hoomd_script import *

system = init.read_xml(filename="{0}")
T_final = {1:.1f}

pot_width = {2:d}
table = pair.table(width=pot_width)

"""


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

    """
    def __init__(self, k, T, state_dir='', traj_file=None, top_file=None):
        self.kT = k * T
        self.state_dir = state_dir

        if not traj_file:
            self.traj_path = os.path.join(state_dir, 'query.dcd')
        # TODO: check if .pdb with same name exists.
        if top_file:
            self.top_path = os.path.join(state_dir, top_file)

    def reload_query_trajectory(self):
        if self.top_path:
            self.traj = md.load(self.traj_path, topology=self.top_path)
        else:
            self.traj = md.load(self.traj_path)

    def save_runscript(self, table_potentials, engine='hoomd',
                       runscript='hoomd_run_template.py'):
        """ """
        header = HOOMD_HEADER.format('start.xml', self.kT, len(R))
        for type1, type2, potential_file in table_potentials:
            command = "table.set_from_file('{0}', '{1}', filename='{2}')\n".format(
               type1, type2, potential_file)
            header += command

        with open(os.path.join(self.state_dir, runscript)) as fh:
            body = ''.join(fh.readlines())

        runscript_file = os.path.join(self.state_dir, 'run.py')
        with open(runscript_file, 'w') as fh:
            fh.write(header)
            fh.write(body)



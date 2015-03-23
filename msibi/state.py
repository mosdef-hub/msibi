import os
from glob import glob

import mdtraj as md


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
            self.traj = md.load(self.traj_path, topology=top_path)
        else:
            self.traj = md.load(traj_path)

    def save_runscript(self, table_potentials, engine='hoomd'):
        """ """

        header = """
        from hoomd_script import *

        system = init.read_xml(filename="{0}")

        pot_width = {1:d}
        table = pair.table(width=pot_width)

        """
        for type1, type2, potential_file in table_potentials:
            command = "table.set_from_file('{0}', '{1}', filename='{2}')\n".format(
               type1, type2, potential_file)
            header += command

        runscript_file = os.path.join(self.state_dir, 'run.py')
        with open(runscript_file, 'w') as fh:
            fh.write(header)
            fh.write(body)



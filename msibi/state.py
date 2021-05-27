import os
import shutil
import warnings
from msibi import MSIBI, utils
from msibi.utils.hoomd_run_template import (HOOMD2_HEADER, HOOMD_TABLE_ENTRY,
    HOOMD_BOND_INIT, HOOMD_BOND_ENTRY, HOOMD_ANGLE_INIT, HOOMD_ANGLE_ENTRY,
    HOOMD_TEMPLATE)

import cmeutils as cme
from cmeutils.structure import gsd_rdf
import gsd
import gsd.hoomd
import mdtraj as md


class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    kT : float, required
        Unitless heat energy (product of Boltzmann's constant and temperature).
    name : str, required
        State name used in creating state directory space and output files.
    traj_file : path to a gsd.hoomd.HOOMDTrajectory file
        The gsd trajectory associated with this state
    alpha : float, optional, default=1.0
        The alpha value used to scaale the weight of this state.
    backup_trajectory : bool, optional, default=False
        True if each query trajectory is backed up (default False)
    """
    def __init__(
        self,
        name,
        kT,
        traj_file,
        alpha=1.0,
        backup_trajectory=False,
    ):
        self.name = name
        self.kT = kT
        self.traj_file = os.path.abspath(traj_file)
        self._opt = None
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be between 0.0 and 1.0")
        self.alpha = float(alpha)
        self.dir = self._setup_dir(name, kT)
        self.query_traj = os.path.join(self.dir, "query.gsd")
        self.backup_trajectory = backup_trajectory

    def reload_query_trajectory(self):
        """Reload the query trajectory."""
        self.traj = md.load(self.traj_file)

    def save_runscript(
        self,
        table_potentials,
        table_width,
        bonds=None,
        angles=None,
        engine="hoomd",
    ):
        """Save the input script for the MD engine."""
        script = list()
        script.append(
                HOOMD2_HEADER.format(self.traj_file, self.kT, table_width)
                )

        for type1, type2, potential_file in table_potentials:
            script.append(HOOMD_TABLE_ENTRY.format(**locals()))

        if bonds is not None:
            script.append(HOOMD_BOND_INIT)
            for bond in bonds:
                name = bond.name
                k = bond._states[self]["k"]
                r0 = bond._states[self]["r0"]
                script.append(HOOMD_BOND_ENTRY.format(**locals()))

        if angles is not None:
            script.append(HOOMD_ANGLE_INIT)
            for angle in angles:
                name = angle.name
                k = angle._states[self]["k"]
                theta = angle._states[self]["theta"]
                script.append(HOOMD_ANGLE_ENTRY.format(**locals()))

        script.append(HOOMD_TEMPLATE)

        runscript_file = os.path.join(self.dir, "run.py")
        with open(runscript_file, "w") as fh:
            fh.writelines(script)

    def _setup_dir(self, name, kT):
        """
        Handle the creation of a state specific directory each time a new
        State() object is created.
        """
        if not os.path.isdir("states"):
            os.mkdir("states")

        dir_name = f"{name}_{kT}"
        try:
            assert not os.path.isdir(os.path.join("states", dir_name))
            os.mkdir(os.path.join("states", dir_name))
        except:
            raise AssertionError("A State object has already "+
                            f"been created with a name of {name} "+
                            f"and a kT of {kT}")
        return os.path.abspath(os.path.join("states", dir_name))


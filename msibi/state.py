import os
import shutil
import warnings
from msibi import MSIBI, utils
from msibi.utils.hoomd_run_template import (HOOMD2_HEADER, HOOMD_TEMPLATE)

import cmeutils as cme
from cmeutils.structure import gsd_rdf
import gsd
import gsd.hoomd


class State(object):
    """A single state used as part of a multistate optimization.

    Parameters
    ----------
    name : str
        State name used in creating state directory space and output files.
    kT : float
        Unitless heat energy (product of Boltzmann's constant and temperature).
    traj_file : path to a gsd.hoomd.HOOMDTrajectory file
        The gsd trajectory associated with this state
    alpha : float, default 1.0
        The alpha value used to scaale the weight of this state.
    backup_trajectory : bool, default False
        True if each query trajectory is backed up

    Attributes
    ----------
    name : str
        State name
    kT : float
        Unitless heat energy (product of Boltzmann's constant and temperature).
    traj_file : path
        Path to the gsd trajectory associated with this state
    alpha : float
        The alpha value used to scaale the weight of this state.
    dir : str
        Path to where the State info with be saved.
    query_traj : str
        Path to the query trajectory.
    backup_trajectory : bool
        True if each query trajectory is backed up

    """
    def __init__(
        self,
        name,
        kT,
        traj_file,
        alpha=1.0,
        backup_trajectory=False,
        _dir=None
    ):
        self.name = name
        self.kT = kT
        self.traj_file = os.path.abspath(traj_file)
        self._opt = None
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be between 0.0 and 1.0")
        self.alpha = float(alpha)
        self.dir = self._setup_dir(name, kT, dir_name=_dir)
        self.query_traj = os.path.join(self.dir, "query.gsd")
        self.backup_trajectory = backup_trajectory

    def _save_runscript(
        self,
        n_steps,
        integrator,
        integrator_kwargs,
        dt,
        gsd_period,
        pairs=None,
        bonds=None,
        angles=None,
    ):
        """Save the input script for the MD engine."""
        script = list()
        script.append(
            HOOMD2_HEADER.format(self.traj_file)
        )
        # TODO Add check that the init scripts are the same for all 
        if pairs is not None and len(pairs) > 0:
            script.append(pairs[0].pair_init)
            for pair in pairs:
                script.append(pair.pair_entry)
         
        if bonds is not None and len(bonds) > 0:
            script.append(bonds[0].bond_init)
            for bond in bonds:
                script.append(bond.bond_entry)

        if angles is not None and len(angles) > 0:
            script.append(angles[0].angle_init)
            for angle in angles:
                script.append(angle_entry)

        integrator_kwargs["kT"] = self.kT
        script.append(HOOMD_TEMPLATE.format(**locals()))

        runscript_file = os.path.join(self.dir, "run.py")
        with open(runscript_file, "w") as fh:
            fh.writelines(script)

    def _setup_dir(self, name, kT, dir_name=None):
        """Create a state directory each time a new State is created."""
        if dir_name is None:
            if not os.path.isdir("states"):
                os.mkdir("states")
            dir_name = os.path.join("states", f"{name}_{kT}")
        else:
            if not os.path.isdir(
                    os.path.join(dir_name, "states")
                    ):
                os.mkdir(os.path.join(dir_name, "states"))
            dir_name = os.path.join(dir_name, "states", f"{name}_{kT}")
        try:
            assert not os.path.isdir(dir_name)
            os.mkdir(dir_name)
        except AssertionError:
            print(f"{dir_name} already exists")
            raise
        return os.path.abspath(dir_name)


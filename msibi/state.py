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
        n_frames,
        alpha=1.0,
        exclude_bonded=True,
        backup_trajectory=False,
        target_frames=None,
        _dir=None
    ):
        self.name = name
        self.kT = kT
        self.traj_file = os.path.abspath(traj_file)
        self._n_frames = n_frames
        self._opt = None
        self._alpha = float(alpha)
        self.dir = self._setup_dir(name, kT, dir_name=_dir)
        self.query_traj = os.path.join(self.dir, "query.gsd")
        self.exclude_bonded = exclude_bonded
        # TODO: Do we want to support saving backup trajs?
        self.backup_trajectory = backup_trajectory

    def __repr__(self):
        return (
                f"{self.__class__}; "
                + f"Name: {self.name}; "
                + f"kT: {self.kT}; "
                + f"Weight: {self.alpha}"
        )

    @property
    def n_frames(self):
        return self._n_frames
    
    @n_frames.setter
    def n_frames(self, value):
        self._n_frames = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    def _save_runscript(
        self,
        n_steps,
        nlist,
        nlist_exclusions,
        integrator,
        integrator_kwargs,
        dt,
        gsd_period,
        pairs=None,
        bonds=None,
        angles=None,
        dihedrals=None,
    ):
        """Save the input script for the MD engine."""
        script = list()
        script.append(
            HOOMD2_HEADER.format(self.traj_file, nlist, nlist_exclusions)
        )
        if pairs is not None and len(pairs) > 0:
            # TODO: Do this check somewhere else? Opt in add_force?
            if len(set([p.force_init for p in pairs])) != 1:
                raise RuntimeError(
                        "Combining different pair potential types "
                        "is not currently supported in MSIBI."
                )
            script.append(pairs[0].force_init)
            for pair in pairs:
                script.append(pair.force_entry)
         
        if bonds is not None and len(bonds) > 0:
            if len(set([b.force_init for b in bonds])) != 1:
                raise RuntimeError(
                        "Combining different bond potential types "
                        "is not currently supported in MSIBI."
                )
            script.append(bonds[0].force_init)
            for bond in bonds:
                script.append(bond.force_entry)

        if angles is not None and len(angles) > 0:
            if len(set([a.force_init for a in angles])) != 1:
                raise RuntimeError(
                        "Combining different angle potential types "
                        "is not currently supported in MSIBI."
                )
            script.append(angles[0].force_init)
            for angle in angles:
                script.append(angle.force_entry)

        if dihedrals is not None and len(dihedrals) > 0:
            if len(set([d.force_init for d in dihedrals])) != 1:
                raise RuntimeError(
                        "Combining different dihedral potential types "
                        "is not currently supported in MSIBI."
                )
            script.append(dihedrals[0].force_init)
            for dihedral in dihedrals:
                script.append(dihedral.force_entry)

        integrator_kwargs["kT"] = self.kT
        # TODO: use locals here or is there a better way?
        script.append(HOOMD_TEMPLATE.format(**locals()))

        runscript_file = os.path.join(self.dir, "run.py")
        with open(runscript_file, "w") as fh:
            fh.writelines("%s\n" % l for l in script)

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

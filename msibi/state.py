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
        self._potential_history = []
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
    
    def _run_simulation(
            self,
            n_steps,
            nlist,
            nlist_exclusions,
            integrator_method,
            method_kwargs,
            dt,
            seed,
            iteration,
            gsd_period,
            pairs=None,
            bonds=None,
            angles=None,
            dihedrals=None 
    ):
        print(f"Starting simulation {iteration} for state {self}")
        device = hoomd.device.auto_select()
        sim = hoomd.simulation.Simulation(device=device)
        with gsd.hoomd.open(self.traj_file, "rb") as traj:
            last_snap = traj[-1]
        sim.create_state_from_snapshot(last_snap)
        
        # Create force objects
        pair_force = None
        for pair in pairs:
            if not pair_force:
                pair_force = getattr(hoomd.md.pair, pair.force_init)
            pair_force.params[pair.name] = dict(**pair.force_entry)

        bond_force = None
        for bond in bonds:
            if not bond_force:
                bond_force = getattr(hoomd.md.bond, bond.force_init)
            bond_force.params[bond.name] = dict(**bond.force_entry)

        angle_force = None
        for angle in angles:
            if not angle_force:
                angle_force = getattr(hoomd.md.angle, angle.force_init)
            angle_force.params[angle.name] = dict(**angle.force_entry)

        dihedral_force = None
        for dih in dihedrals:
            if not dihedral_force:
                dihedral_force = getattr(hoomd.md.dihedral, dihedral.force_init)
            dihedral_force.params[dihedral.name] = dict(**dihedral.force_entry)

        # Create integrator and integration method
        #TODO: Set kT in method_kwargs
        forces = [pair_force, bond_force, angle_force, dihedral_force]
        integrator = hoomd.md.Integrator(dt=dt) 
        integrator.forces = [f for f in forces if f] 
        method = getattr(hoomd.md.methods, integrator_method)
        integrator.methods.append(method(**method_kwargs))
        sim.operations.add(integrator)

        #Create GSD writer
        gsd_writer = hoomd.write.GSD(
                filename=self.query_traj,
                trigger=hoomd.trigger.Periodic(int(gsd_period)),
                mode="wb",
        )
        sim.operations.writers.append(gsd_writer)

        # Run simulation
        sim.run(n_steps)
        print(f"Finished simulation {iteration} for state {self}")

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

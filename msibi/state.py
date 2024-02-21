import os
import shutil
import warnings


import cmeutils as cme
from cmeutils.structure import gsd_rdf
import gsd
import gsd.hoomd
import hoomd
from msibi import MSIBI, utils
from msibi.utils.hoomd_run_template import (HOOMD2_HEADER, HOOMD_TEMPLATE)


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
            thermostat,
            method_kwargs,
            dt,
            seed,
            r_cut,
            iteration,
            gsd_period,
            pairs=None,
            bonds=None,
            angles=None,
            dihedrals=None,
            backup_trajectories=False
    ):
        device = hoomd.device.auto_select()
        sim = hoomd.simulation.Simulation(device=device)
        print(f"Starting simulation {iteration} for state {self}")
        print(f"Running on device {device}")

        with gsd.hoomd.open(self.traj_file, "r") as traj:
            last_snap = traj[-1]
        sim.create_state_from_snapshot(last_snap)
        
        nlist = getattr(hoomd.md.nlist, nlist)
        thermostat = getattr(hoomd.md.methods.thermostats, thermostat)
        # Create pair objects
        pair_force = None
        for pair in pairs:
            if not pair_force: # Only create hoomd.md.pair obj once
                hoomd_pair_force = getattr(hoomd.md.pair, pair.force_init)
                if pair.force_init == "Table":
                    pair_force = hoomd_pair_force(width=pair.nbins)
                else:
                    pair_force = hoomd_pair_force(
                            nlist=nlist(buffer=20, exclusions=nlist_exclusions),
                            default_r_cut=r_cut
                    ) 
            param_name = (pair.name[0], pair.name[-1]) # Can't use pair.name
            pair_force.params[param_name] = pair.force_entry

        # Create bond objects
        bond_force = None
        for bond in bonds:
            if not bond_force:
                hoomd_bond_force = getattr(hoomd.md.bond, bond.force_init)
                if bond.force_init == "Table":
                    bond_force = hoomd_bond_force(width=bond.nbins)
                else:
                    bond_force = hoomd_bond_force()
            bond_force.params[bond.name] = bond.force_entry

        # Create angle objects
        angle_force = None
        for angle in angles:
            if not angle_force:
                hoomd_angle_force = getattr(hoomd.md.angle, angle.force_init)
                if angle.force_init == "Table":
                    angle_force = hoomd_angle_force(width=angle.nbins)
                else:
                    angle_force = hoomd_angle_force()
            angle_force.params[angle.name] = angle.force_entry

        # Create dihedral objects
        dihedral_force = None
        for dih in dihedrals:
            if not dihedral_force:
                hoomd_dihedral_force = getattr(
                        hoomd.md.dihedral, dih.force_init
                )
                if dih.force_init == "Table":
                    dihedral_force = hoomd_dihedral_force(width=dih.nbins)
                else:
                    dihedral_force = hoomd_dihedral_force()
            dihedral_force.params[dih.name] = dih.force_entry

        # Create integrator and integration method
        #TODO: Set kT in method_kwargs
        forces = [pair_force, bond_force, angle_force, dihedral_force]
        integrator = hoomd.md.Integrator(dt=dt) 
        integrator.forces = [f for f in forces if f] # Filter out None 
        method = getattr(hoomd.md.methods, integrator_method)
        integrator.methods.append(
                method(filter=hoomd.filter.All(), kT=self.kT, **method_kwargs)
        )
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
        if backup_trajectories:
            pass #TODO: shutil copy traj file
        print(f"Finished simulation {iteration} for state {self}")
        print()

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

import os
import pickle
import shutil

import hoomd
import numpy as np

import msibi


class MSIBI(object):
    """
    Management class for orchestrating an MSIBI optimization.

    Parameters
    ----------
    nlist : str, required
        The type of hoomd neighbor list to use.
        When optimizing bonded potentials, using hoomd.md.nlist.tree
        may work best for single chain, low density simulations
        When optimizing pair potentials hoomd.md.nlist.cell
        may work best
    integrator_method : str, required
        The integrator_method to use in the query simulation.
    integrator_kwargs : dict, required
        The args and their values required by the integrator chosen
    dt : float, required
        The time step delta
    gsd_period : int, required
        The number of frames between snapshots written to query.gsd
    n_steps : int, required
        How many steps to run the query simulations
    r_cut : float, optional, default 0
        Set the r_cut value to use in pair interactions.
        Leave as zero if pair interactions aren't being used.
    nlist_exclusions : list of str, optional, default ["1-2", "1-3"]
        Sets the pair exclusions used during the optimization simulations
    seed : int, optional, default 42
        Random seed to use during the simulation
    backup_trajectories : bool, optional, default False
        If False, the query simulation trajectories are
        overwritten during each iteraiton.
        If True, the query simulations are saved for
        each iteration.

    Attributes
    ----------
    states : list of msibi.state.State
        All states to be used in the optimization procedure.
    pairs : list of msibi.pair.Pair
        All pairs to be used in the optimization procedure.
    bonds : list of msibi.bonds.Bond
        All bonds to be used in the optimization procedure.
    angles : list of msibi.bonds.Angle
        All angles to be used in the optimization procedure.
    dihedrals : list of msibi.bonds.Dihedral
        All dihedrals to be used in the optimization procedure.

    Methods
    -------
    add_state(msibi.state.state)
        Add a state point to be included in optimizing forces.
    add_force(msibi.forces.Force)
        Add the required interaction objects. See forces.py
    run_optimization(n_iterations, n_steps, backup_trajectories)
        Performs iterations of query simulations and potential updates
        resulting in a final optimized potential.
    pickle_forces()
        Saves a pickle file containing a list of Hoomd force objects
        as they existed in the most recent optimization run.

    """

    def __init__(
            self,
            nlist,
            integrator_method,
            thermostat,
            method_kwargs,
            thermostat_kwargs,
            dt: float,
            gsd_period: int,
            r_cut,
            nlist_exclusions: list[str]=["bond", "angle"],
            seed: int=42,
    ):
        if nlist not in ["Cell", "Tree", "Stencil"]:
            raise ValueError(f"{nlist} is not a valid neighbor list in Hoomd")
        self.nlist = nlist
        self.integrator_method = integrator_method
        self.thermostat = thermostat
        self.method_kwargs = method_kwargs
        self.thermostat_kwargs = thermostat_kwargs
        self.dt = dt
        self.gsd_period = gsd_period
        self.r_cut = r_cut
        self.seed = seed
        self.nlist_exclusions = nlist_exclusions
        self.n_iterations = 0
        self.states = []
        self.forces = []
        self._optimize_forces = []

    def add_state(self, state: msibi.state.State) -> None:
        """Add a state point to MSIBI.states.

        Parameters
        ----------
        state : msibi.state.State, required
            Instance of msibi.state.State
        """
        #TODO: Do we need this?
        state._opt = self
        self.states.append(state)

    def add_force(self, force: msibi.forces.Force) -> None:
        """Add a force to be included in the query simulations.

        Parameters
        ----------
        force : msibi.forces.Force, required
            Instance of msibi.forces.Force

        Notes
        -----
        Only one type of force can be optimized at a time.
        Forces not set to be optimized are held fixed during query simulations.

        """
        self.forces.append(force)
        if force.optimize:
            self._add_optimize_force(force)
        for state in self.states:
            force._add_state(state)

    @property
    def bonds(self):
        """All instances of msibi.forces.Bond that have been added."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Bond)]

    @property
    def angles(self):
        """All instances of msibi.forces.Angle that have been added."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Angle)]

    @property
    def pairs(self):
        """All instances of msibi.forces.Pair that have been added."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Pair)]

    @property
    def dihedrals(self):
        """All instances of msibi.forces.Dihedral that have been added."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Dihedral)]

    def _add_optimize_force(self, force):
        if not all(
                [isinstance(force, f.__class__) for f in self._optimize_forces]
        ):
            raise RuntimeError(
                    "Only one type of force (i.e. Bonds, Angles, Pairs, etc) "
                    "can be set to optimize at a time."
            )
        self._optimize_forces.append(force)

    def run_optimization(
            self,
            n_steps: int,
            n_iterations: int,
            backup_trajectories: bool=False,
            _dir=None
    ) -> None:
        """Runs query simulations and performs MSIBI
        on the potentials set to be optimized.

        Parameters
        ----------
        n_steps : int, required
            Number of simulation steps during each iteration.
        n_iterations : int, required
            Number of MSIBI update iterations.
        backup_trajectories : bool, optional default False
            If True, copies of the query simulation trajectories
            are saved in their respective msibi.state.State directory.

        """
        for n in range(n_iterations):
            print(f"---Optimization: {n+1} of {n_iterations}---")
            forces = self._build_objects()
            for state in self.states:
                state._run_simulation(
                    n_steps=n_steps,
                    forces=forces,
                    integrator_method=self.integrator_method,
                    method_kwargs=self.method_kwargs,
                    thermostat=self.thermostat,
                    thermostat_kwargs=self.thermostat_kwargs,
                    dt=self.dt,
                    seed=self.seed,
                    iteration=self.n_iterations,
                    gsd_period=self.gsd_period,
                    backup_trajectories=backup_trajectories
                )
            self._update_potentials()
            self.n_iterations += 1

    def pickle_forces(self, file_path: str) -> None:
        """Save the Hoomd objects for all forces to a single pickle file.

        Parameters
        ----------
        file_path : str, required
            The path and file name for the pickle file.

        Notes
        -----
        Use this method as a convienent way to save and use the final
        set of forces in your own Hoomd-Blue script, or with
        flowerMD (https://github.com/cmelab/flowerMD).

        """
        forces = self._build_force_objects()
        if len(forces) == 0:
            raise RuntimeError(
                    "No forces have been created yet. See MSIBI.add_force()"
            )
        f = open(file_path, "wb")
        pickle.dump(forces, f)
    
    def _build_foce_objects(self) -> list:
        """Creates force objects for query simulations."""
        nlist = getattr(hoomd.md.nlist, self.nlist)
        # Create pair objects
        pair_force = None
        for pair in self.pairs:
            if not pair_force: # Only create hoomd.md.pair obj once
                hoomd_pair_force = getattr(hoomd.md.pair, pair.force_init)
                if pair.force_init == "Table":
                    pair_force = hoomd_pair_force(width=pair.nbins)
                else:
                    pair_force = hoomd_pair_force(
                            nlist=nlist(
                                buffer=20,
                                exclusions=self.nlist_exclusions
                            ),
                            default_r_cut=self.r_cut
                    )
            #TODO: this won't work if the pair types aren't single letters
            param_name = (pair.name[0], pair.name[-1]) # Can't use pair.name
            if pair.format == "table":
                pair_force.params[pair._pair_name] = pair._table_entry()
            else:
                pair_force.params[pair._pair_name] = pair.force_entry
        # Create bond objects
        bond_force = None
        for bond in self.bonds:
            if not bond_force:
                hoomd_bond_force = getattr(hoomd.md.bond, bond.force_init)
                if bond.force_init == "Table":
                    bond_force = hoomd_bond_force(width=bond.nbins + 1)
                else:
                    bond_force = hoomd_bond_force()
            if bond.format == "table":
                bond_force.params[bond.name] = bond._table_entry()
            else:
                bond_force.params[bond.name] = bond.force_entry
        # Create angle objects
        angle_force = None
        for angle in self.angles:
            if not angle_force:
                hoomd_angle_force = getattr(hoomd.md.angle, angle.force_init)
                if angle.force_init == "Table":
                    angle_force = hoomd_angle_force(width=angle.nbins + 1)
                else:
                    angle_force = hoomd_angle_force()
            if angle.format == "table":
                angle_force.params[angle.name] = angle._table_entry()
            else:
                angle_force.params[angle.name] = angle.force_entry
        # Create dihedral objects
        dihedral_force = None
        for dih in self.dihedrals:
            if not dihedral_force:
                hoomd_dihedral_force = getattr(
                        hoomd.md.dihedral, dih.force_init
                )
                if dih.force_init == "Table":
                    dihedral_force = hoomd_dihedral_force(width=dih.nbins + 1)
                else:
                    dihedral_force = hoomd_dihedral_force()
            if dih.format == "table":
                dihedral_force.params[dih.name] = dih._table_entry()
            else:
                dihedral_force.params[dih.name] = dih.force_entry
        forces = [pair_force, bond_force, angle_force, dihedral_force]
        return [f for f in forces if f] # Filter out any None values

    def _update_potentials(self) -> None:
        """Update the potentials for the potentials to be optimized."""
        for force in self._optimize_forces:
            self._recompute_distribution(force)
            force._update_potential()

    def _recompute_distribution(self, force: msibi.forces.Force) -> None:
        """Recompute the current distribution of bond lengths or angles"""
        for state in self.states:
            force._compute_current_distribution(state)
            force._save_current_distribution(
                    state,
                    iteration=self.n_iterations
            )
            print("Force: {0}, State: {1}, Iteration: {2}: {3:f}".format(
                    force.name,
                    state.name,
                    self.n_iterations,
                    force._states[state]["f_fit"][self.n_iterations]
                )
            )
            print()

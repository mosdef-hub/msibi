import os
import shutil

import numpy as np

import msibi


class MSIBI(object):
    """Management class for orchestrating an MSIBI optimization.

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
    add_force(msibi.forces.Force)
        Add the required interaction objects. See forces.py
    run_optimization(n_iterations, backup_trajectories)
        Performs iterations of query simulations and potential updates
        resulting in a final optimized potential.

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
            for state in self.states:
                state._run_simulation(
                    n_steps=n_steps,
                    nlist=self.nlist,
                    nlist_exclusions=self.nlist_exclusions,
                    integrator_method=self.integrator_method,
                    method_kwargs=self.method_kwargs,
                    thermostat=self.thermostat,
                    thermostat_kwargs=self.thermostat_kwargs,
                    dt=self.dt,
                    r_cut=self.r_cut,
                    seed=self.seed,
                    iteration=self.n_iterations,
                    gsd_period=self.gsd_period,
                    pairs=self.pairs,
                    bonds=self.bonds,
                    angles=self.angles,
                    dihedrals=self.dihedrals,
                    backup_trajectories=backup_trajectories
                )
            self._update_potentials()
            self.n_iterations += 1

    def pickle_forcefield(self, file_path: str) -> None:
        """
        Save the Hoomd objects for all forces to a single pickle file.

        Parameters
        ----------
        file_path : str, required
            The path and file name for the pickle file.

        Notes
        -----
        Use this method as a convienent way to use the final
        set of forces in your own Hoomd-Blue script, or with
        flowerMD (https://github.com/cmelab/flowerMD)

        """


    def _update_potentials(self):
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

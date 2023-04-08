import os
import shutil

import numpy as np

import msibi
from msibi.potentials import pair_tail_correction, save_table_potential
from msibi.utils.smoothing import savitzky_golay
from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import run_query_simulations


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
    add_state(state)
    add_force(msibi.forces.Force)
        Add the required interaction objects. See forces.py

    optimize_bonds(n_iterations)
        Calculates the target bond length distributions for each Bond
        in MSIBI.bonds and optimizes the bonding potential.

    """
    def __init__(
            self,
            nlist,
            integrator_method,
            method_kwargs,
            dt,
            gsd_period,
            n_steps,
            r_cut=0,
            nlist_exclusions=["bond", "angle"],
            seed=42,
            backup_trajectories=False
    ):
        if integrator_method == "NVE":
            raise ValueError("The NVE ensemble is not supported with MSIBI")

        if nlist not in ["Cell", "Tree", "Stencil"]:
            raise ValueError(f"{nlist} is not a valid neighbor list in Hoomd")

        self.nlist = nlist 
        self.integrator_method = integrator_method
        self.method_kwargs = method_kwargs
        self.dt = dt
        self.gsd_period = gsd_period
        self.n_steps = n_steps
        self.r_cut = r_cut
        self.seed = seed
        self.nlist_exclusions = nlist_exclusions
        self.backup_trajectories = backup_trajectories
        self.states = []
        self.forces = []
        self._optimize_forces = []

    def add_state(self, state):
        """"""
        state._opt = self
        self.states.append(state)

    def add_force(self, force):
        """"""
        self.forces.append(force)
        if force.optimize:
            self._add_optimize_force(force)
        for state in self.states:
            force._add_state(state)

    @property
    def bonds(self):
        return [f for f in self.forces if isinstance(f, msibi.forces.Bond)]

    @property
    def angles(self):
        return [f for f in self.forces if isinstance(f, msibi.forces.Angle)]

    @property
    def pairs(self):
        return [f for f in self.forces if isinstance(f, msibi.forces.Pair)]

    @property
    def dihedrals(self):
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

    def run_optimization(self, n_iterations, _dir=None):
        """Runs MSIBI on the potentials set to be optimized.

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        """
        self._initialize(potentials_dir=_dir)
        for n in range(n_iterations):
            print(f"---Optimization: {n+1} of {n_iterations}---")
            for state in self.states:
                state._run_simulation(
                    n_steps=self.n_steps,
                    nlist=self.nlist,
                    nlist_exclusions=self.nlist_exclusions,
                    integrator_method=self.integrator_method,
                    method_kwargs=self.method_kwargs,
                    dt=self.dt,
                    r_cut=self.r_cut,
                    seed=self.seed,
                    iteration=n+1,
                    gsd_period=self.gsd_period,
                    pairs=self.pairs,
                    bonds=self.bonds,
                    angles=self.angles,
                    dihedrals=self.dihedrals,
                    backup_trajectories=self.backup_trajectories
                )
            #TODO: Make sure this working
            self._update_potentials(n)

        # After MSIBI iterations are done: What are we doing?
        # Save final potentials to a text file
        # Skip smoothing here?
        for force in self._optimize_forces:
            if force.smoothing_window and force.smoothing_order:
                smoothed_pot = savitzky_golay(
                        y=force.potential,
                        window_size=force.smoothing_window,
                        order=force.smoothing_order
                )
            else:
                smoothed_pot = force.potential
            file_name = f"{force.name}_final.txt"
            save_table_potential(
                    potential=smoothed_pot,
                    r=force.x_range,
                    dr=force.dx,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def _update_potentials(self, iteration):
        """Update the potentials for the potentials to be optimized."""
        for force in self._optimize_forces:
            self._recompute_distribution(force, iteration)
            force._update_potential()
            save_table_potential(
                    force.potential,
                    force.x_range,
                    force.dx,
                    iteration,
                    force._potential_file
            )

    def _recompute_distribution(self, force, iteration):
        """Recompute the current distribution of bond lengths or angles"""
        for state in self.states:
            force._compute_current_distribution(state)
            force._save_current_distribution(state, iteration=iteration)
            print("{0}, State: {1}, Iteration: {2}: {3:f}".format(
                    force.name,
                    state.name,
                    iteration + 1,
                    force._states[state]["f_fit"][iteration]
                )
            )
            print()

    def _initialize(self, potentials_dir):
        """Create initial table potentials and the simulation input scripts.

        Parameters
        ----------
        potentials_dir : path, default None
            Directory to store potential files. If None is given, a "potentials"
            folder in the current working directory is used.

        """
        if potentials_dir is None:
            self.potentials_dir = os.path.join(os.getcwd(), "potentials")
        else:
            self.potentials_dir = potentials_dir

        if not os.path.isdir(self.potentials_dir):
            os.mkdir(self.potentials_dir)
        for force in self.forces:
            if force.format == "table" and force.optimize:
                potential_file = os.path.join(
                    self.potentials_dir, f"{force.name}.txt"
                )
                force._potential_file = potential_file
                save_table_potential(
                        force.potential,
                        force.x_range,
                        force.dx,
                        0,
                        force._potential_file
                )

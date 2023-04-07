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
    integrator : str, required 
        The integrator to use in the query simulation.
        See hoomd-blue.readthedocs.io/en/v2.9.6/module-md-integrate.html
    integrator_kwargs : dict, required 
        The args and their values required by the integrator chosen
    dt : float, required 
        The time step delta
    gsd_period : int, required 
        The number of frames between snapshots written to query.gsd
    n_steps : int, required 
        How many steps to run the query simulations
    nlist_exclusions : list of str, optional, default ["1-2", "1-3"]
        Sets the pair exclusions used during the optimization simulations

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
            integrator,
            integrator_kwargs,
            dt,
            gsd_period,
            n_steps,
            nlist_exclusions=["1-2", "1-3"],
    ):
        if integrator == "hoomd.md.integrate.nve":
            raise ValueError("The NVE ensemble is not supported with MSIBI")

        assert nlist in [
                "hoomd.md.nlist.cell",
                "hoomd.md.nlist.tree",
                "hoomd.md.nlist.stencil"
        ], "Enter a valid Hoomd neighbor list type"

        self.nlist = nlist 
        self.integrator = integrator
        self.integrator_kwargs = integrator_kwargs
        self.dt = dt
        self.gsd_period = gsd_period
        self.n_steps = n_steps
        self.nlist_exclusions = nlist_exclusions
        # Store all of the needed interaction objects
        self.states = []
        self.forces = []
        self._optimize_forces = []

    def add_state(self, state):
        state._opt = self
        self.states.append(state)

    def add_force(self, force):
        self.forces.append(force)
        if force.optimize:
            self._add_optimize_force(force)
        for state in self.states:
            force._add_state(state)

    def _add_optimize_force(self, force):
        if not all(
                [isinstance(force, f.__class__) for f in self._optimize_forces]
        ):
            raise RuntimeError(
                    "Only one type of force (i.e. Bonds, Angles, Pairs, etc) "
                    "Can be set to optimize."
            )
        self._optimize_forces.append(force)

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
            run_query_simulations(self.states)
            self._update_potentials(n)
        for force in self._optimize_forces:
            if not smooth_pot: 
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

    def optimize_bonds(
            self,
            n_iterations,
            smooth_dist=True,
            smooth_pot=True,
            _dir=None
    ):
        """Optimize the bond potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        smooth_dist : bool, default True
            If True, the target distribution is smoothed
        smooth_pot : bool, default True
            If True, the potential is smoothed between iterations

        """
        self.optimization = "bonds"
        self.smooth_dist = smooth_dist
        self._initialize(potentials_dir=_dir)
        # Run the optimization iterations:
        for n in range(n_iterations):
            print(f"---Bond Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n)
        # Save final potential to a seprate file
        # If not already smoothing the potential, smooth the final output
        for bond in self.bonds:
            if not smooth_pot: 
                smoothed_pot = savitzky_golay(
                        y=bond.potential, window_size=smoothing_window, order=1
                )
            else:
                smoothed_pot = bond.potential
            file_name = f"{bond.name}_final.txt"
            save_table_potential(
                    potential=smoothed_pot,
                    r=bond.x_range,
                    dr=bond.dx,
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
        #TODO: Fix this stuff to work with single list of force objects
        #TODO: Set optimization attribute for MSIBI class?
        for force in self.forces:
            if force.format == "table" and force.optimize:
                potential_file = os.path.join(
                    self.potentials_dir, f"pair_pot_{force.name}.txt"
                )
                force.update_potential_file(potential_file)
                V = pair_tail_correction(
                        pair.x_range, pair.potential, pair.r_switch
                )
                pair.potential = V
                if self.optimization == "pairs":
                    iteration = 0
                else:
                    iteration = None
                save_table_potential(
                        pair.potential,
                        pair.x_range,
                        pair.dx,
                        iteration,
                        pair._potential_file
                )

        for bond in self.bonds: #TODO: Remind myself what we are doing here
            if bond.format == "table" and bond._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"bond_pot_{bond.name}.txt"
                )
                bond.update_potential_file(potential_file)

                if self.optimization == "bonds":
                    iteration = 0
                else:
                    iteration = None

                save_table_potential(
                        bond.potential,
                        bond.x_range,
                        bond.dx,
                        iteration,
                        bond._potential_file
                )

        for angle in self.angles:
            if angle.format == "table" and angle._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"angle_pot_{angle.name}.txt"
                )
                angle.update_potential_file(potential_file)
            elif angle.format == "table" and angle._potential_file != "":
                potential_file = os.path.join(self.potentials_dir, f"angle_pot_{angle.name}")
                # What is this doing, not done for pairs or bonds
                shutil.copyfile(angle._potential_file, potential_file)
                angle.update_potential_file(potential_file)

            if self.optimization == "angles":
                iteration = 0
            else:
                iteration = None

            save_table_potential(
                    angle.potential,
                    angle.x_range,
                    angle.dx,
                    iteration,
                    angle._potential_file
            )

        for dihedral in self.dihedrals:
            if dihedral.format == "table" and dihedral._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"dihedral_pot_{dihedral.name}.txt"
                )
                dihedral.update_potential_file(potential_file)

                if self.optimization == "dihedrals":
                    iteration = 0
                else:
                    iteration = None

                save_table_potential(
                        dihedral.potential,
                        dihedral.x_range,
                        dihedral.dx,
                        iteration,
                        dihedral._potential_file
                )

        for state in self.states:
            state._save_runscript(
                n_steps=int(self.n_steps),
                nlist=self.nlist,
                nlist_exclusions=self.nlist_exclusions,
                integrator=self.integrator,
                integrator_kwargs=self.integrator_kwargs,
                dt=self.dt,
                gsd_period=self.gsd_period,
                pairs=self.pairs,
                bonds=self.bonds,
                angles=self.angles,
                dihedrals=self.dihedrals,
            )

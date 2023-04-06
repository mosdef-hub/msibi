import os
import shutil

import numpy as np

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
    add_bond(bond)
    add_angle(angle)
    add_dihedral(dihedral)
        Add the required interaction objects. See Pair.py and Bonds.py

    optimize_bonds(n_iterations)
        Calculates the target bond length distributions for each Bond
        in MSIBI.bonds and optimizes the bonding potential.

    optimize_angles(n_iterations)
        Calcualtes the target bond angle distribution for each Bond
        in MSIBI.angles and optimizes the angle potential.

    optimize_pairs(smooth_rdfs, r_switch, n_iterations)
        Calculates the target RDF for each Pair in MSIBI.pairs
        and optimizes the pair potential.

    optimize_dihedrals(n_iterations)
        Calculates the target bond dihedral distributions for each Pair 
        in MSIBI.dihedrals and optimizes the dihedral potential.

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
        self.pairs = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []

    def add_state(self, state):
        state._opt = self
        self.states.append(state)

    def add_pair(self, pair):
        self.pairs.append(pair)
        for state in self.states:
            pair._add_state(state)

    def add_bond(self, bond):
        self.bonds.append(bond)
        for state in self.states:
            bond._add_state(state)

    def add_angle(self, angle):
        self.angles.append(angle)

    def add_dihedral(self, dihedral):
        self.dihedrals.append(dihedral)

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
        #self._add_states()
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

    def optimize_angles(
            self,
            n_iterations,
            smooth_dist=True,
            smooth_pot=True,
            smoothing_window=5,
            _dir=None
    ):
        """Optimize the bond angle potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        smooth_dist : bool, default True
            If True, the target distribution is smoothed 
        smooth_pot : bool, default True
            If True, the potential is smoothed between iterations

        """
        self.optimization = "angles"
        self.smooth_dist = smooth_dist
        self._add_states(smoothing_window)
        self._initialize(potentials_dir=_dir)

        for n in range(n_iterations):
            print(f"---Angle Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n, smooth_pot, smoothing_window)
        # Save final potential to a seprate file
        # If not already smoothing the potential, smooth the final output
        for angle in self.angles:
            if not smooth_pot:
                smoothed_pot = savitzky_golay(
                        y=angle.potential, window_size=smoothing_window, order=1
                )
            else:
                smoothed_pot = angle.potential
            file_name = f"{angle.name}_final.txt"
            save_table_potential(
                    potential=angle.potential,
                    r=angle.x_range,
                    dr=angle.dx,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def optimize_pairs(
        self,
        n_iterations,
        smooth_rdfs=True,
        smooth_pot=False,
        smoothing_window=9,
        r_switch=None,
        _dir=None
    ):
        """Optimize the pair potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        smooth_rdfs : bool, default=True
            Set to True to perform smoothing (Savitzky Golay) on the target
            and iterative RDFs.
        smooth_pot : bool, default True
            If True, the potential is smoothed between iterations
        r_switch : float, optional, default=None
            The distance after which a tail correction is applied.
            If None, then Pair.r_range[-5] is used.

        """
        self.optimization = "pairs"
        self.smooth_rdfs = smooth_rdfs
        for pair in self.pairs:
            if r_switch is None:
                pair.r_switch = pair.r_range[-5]
            else:
                pair.r_switch = r_switch

        self._add_states(smoothing_window)
        self._initialize(potentials_dir=_dir)

        for n in range(n_iterations):
            print(f"---Pair Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n, smooth_pot, smoothing_window)

        # Save final potential to a seprate file
        # If not already smoothing the potential, smooth the final output
        for pair in self.pairs:
            if not smooth_pot:
                smoothed_pot = savitzky_golay(
                        y=pair.potential, window_size=smoothing_window, order=1
                )
            else:
                smoothed_pot = pair.potential
            file_name = f"{pair.name}_final.txt"
            save_table_potential(
                    potential=smoothed_pot,
                    r=pair.x_range,
                    dr=pair.dx,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def optimize_dihedrals(
            self,
            n_iterations,
            smooth_dist=True,
            smooth_pot=False,
            smoothing_window=7,
            _dir=None
    ):
        """Optimize the bond dihedral potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        smooth_dist : bool, default True
            If True, the target distribution is smoothed
        smooth_pot : bool, default True
            If True, the potential is smoothed between iterations

        """
        self.optimization = "dihedrals"
        self.smooth_dist = smooth_dist
        self._add_states(smoothing_window)
        self._initialize(potentials_dir=_dir)

        for n in range(n_iterations):
            print(f"---Dihedral Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n, smooth_pot, smoothing_window)
        # Save final potential
        for dihedral in self.dihedrals:
            smoothed_pot = savitzky_golay(
                    dihedral.potential, window_size=5, order=1
            )
            file_name = f"{dihedral.name}_smoothed.txt"
            save_table_potential(
                    potential=smoothed_pot,
                    r=dihedral.x_range,
                    dr=dihedral.dx,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def _add_states(self):
        """Add State objects to Pairs, Bonds, and Angles.
        Required step before optimization runs can begin.

        """
        for pair in self.pairs:
            for state in self.states:
                pair._add_state(state)

        for bond in self.bonds:
            for state in self.states:
                bond._add_state(state)

        for angle in self.angles:
            for state in self.states:
                angle._add_state(state)

        for dihedral in self.dihedrals:
            for state in self.states:
                dihedral._add_state(state)

    def _update_potentials(self, iteration):
        """Update the potentials for the potentials to be optimized."""
        if self.optimization == "pairs":
            for pair in self.pairs:
                self._recompute_distribution(pair, iteration)
                pair._update_potential()
                save_table_potential(
                        pair.potential,
                        pair.x_range,
                        pair.dx,
                        iteration,
                        pair._potential_file
                )

        elif self.optimization == "bonds":
            for bond in self.bonds:
                self._recompute_distribution(bond, iteration)
                bond._update_potential()
                save_table_potential(
                        bond.potential,
                        bond.x_range,
                        bond.dx,
                        iteration,
                        bond._potential_file
                )

        elif self.optimization == "angles":
            for angle in self.angles:
                self._recompute_distribution(angle, iteration)
                angle._update_potential()
                save_table_potential(
                        angle.potential,
                        angle.x_range,
                        angle.dx,
                        iteration,
                        angle._potential_file
                )

        elif self.optimization == "dihedrals":
            for dihedral in self.dihedrals:
                self._recompute_distribution(dihedral, iteration)
                dihedral._update_potential()
                save_table_potential(
                        dihedral.potential,
                        dihedral.x_range,
                        dihedral.dx,
                        iteration,
                        dihedral._potential_file
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

        for pair in self.pairs:
            if pair.force_type == "table" and self.optimization == "pairs":
                potential_file = os.path.join(
                    self.potentials_dir, f"pair_pot_{pair.name}.txt"
                )
                pair.update_potential_file(potential_file)
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
            if bond.force_type == "table" and bond._potential_file == "":
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
            if angle.force_type == "table" and angle._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"angle_pot_{angle.name}.txt"
                )
                angle.update_potential_file(potential_file)
            elif angle.angle_type == "table" and angle._potential_file != "":
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
            if dihedral.force_type == "table" and dihedral._potential_file == "":
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

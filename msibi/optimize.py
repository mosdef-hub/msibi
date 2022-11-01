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

    optimize_bonds(n_iterations, start_iteration)
        Calculates the target bond length distributions for each Bond
        in MSIBI.bonds and optimizes the bonding potential.

    optimize_angles(n_iterations, start_iteration)
        Calcualtes the target bond angle distribution for each Bond
        in MSIBI.angles and optimizes the angle potential.

    optimize_pairs(smooth_rdfs, r_switch, n_iterations)
        Calculates the target RDF for each Pair in MSIBI.pairs
        and optimizes the pair potential.

    optimize_dihedrals(n_iterations, start_iteration)
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

    def add_bond(self, bond):
        self.bonds.append(bond)

    def add_angle(self, angle):
        self.angles.append(angle)

    def add_dihedral(self, dihedral):
        self.dihedrals.append(dihedral)

    def optimize_bonds(
            self,
            n_iterations,
            start_iteration=0,
            smooth=True,
            smooth_pot=True,
            smoothing_window=5,
            _dir=None
    ):
        """Optimize the bond potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        start_iteration : int, default 0
            Start optimization at start_iteration, useful for restarting.
        smooth : bool, default True
            If True, the target distribution is smoothed using a 
            Savitzky-Golay filter

        """
        self.optimization = "bonds"
        self.smooth_dist = smooth
        self._add_states(smoothing_window)
        self._initialize(potentials_dir=_dir)

        for n in range(start_iteration + n_iterations):
            print(f"---Bond Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n, smooth_pot, smoothing_window)
        # Save final potential
        for bond in self.bonds:
            smoothed_pot = savitzky_golay(
                    y=bond.potential, window_size=smoothing_window, order=1
            )
            file_name = f"{bond.name}_smoothed.txt"
            save_table_potential(
                    potential=smoothed_pot,
                    r=bond.l_range,
                    dr=bond.dl,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def optimize_angles(
            self,
            n_iterations,
            start_iteration=0,
            smooth=True,
            smooth_pot=True,
            smoothing_window=5,
            _dir=None
    ):
        """Optimize the bond angle potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        start_iteration : int, default 0
            Start optimization at start_iteration, useful for restarting.
        smooth : bool, default True
            If True, the target distribution is smoothed using a 
            Savitzky-Golay filter

        """
        self.optimization = "angles"
        self.smooth_dist = smooth
        self._add_states(smoothing_window)
        self._initialize(potentials_dir=_dir)

        for n in range(start_iteration + n_iterations):
            print(f"---Angle Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n, smooth_pot, smoothing_window)
        # Save final potential
        for angle in self.angles:
            smoothed_pot = savitzky_golay(
                    y=angle.potential, window_size=smoothing_window, order=1
            )
            file_name = f"{angle.name}_smoothed.txt"
            save_table_potential(
                    potential=angle.potential,
                    r=angle.theta_range,
                    dr=angle.dtheta,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def optimize_pairs(
        self,
        n_iterations,
        start_iteration=0,
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
        start_iteration : int, default 0
            Start optimization at start_iteration, useful for restarting.
        smooth_rdfs : bool, default=True
            Set to True to perform smoothing (Savitzky Golay) on the target
            and iterative RDFs.
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

        for n in range(start_iteration + n_iterations):
            print(f"---Pair Optimization: {n+1} of {n_iterations}---")
            run_query_simulations(self.states)
            self._update_potentials(n, smooth_pot, smoothing_window)

        for pair in self.pairs:
            smoothed_pot = savitzky_golay(
                    y=pair.potential, window_size=smoothing_window, order=1
            )
            file_name = f"{pair.name}_smoothed.txt"
            save_table_potential(
                    potential=smoothed_pot,
                    r=pair.r_range,
                    dr=pair.dr,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def optimize_dihedrals(
            self,
            n_iterations,
            start_iteration=0,
            smooth=True,
            smooth_pot=False,
            smoothing_window=7,
            _dir=None
    ):
        """Optimize the bond dihedral potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        start_iteration : int, default 0
            Start optimization at start_iteration, useful for restarting.
        smooth : bool, default True
            If True, the target distribution is smoothed using a 
            Savitzky-Golay filter

        """
        self.optimization = "dihedrals"
        self.smooth_dist = smooth
        self._add_states(smoothing_window)
        self._initialize(potentials_dir=_dir)

        for n in range(start_iteration + n_iterations):
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
                    r=dihedral.phi_range,
                    dr=dihedral.dphi,
                    iteration=None,
                    potential_file=os.path.join(self.potentials_dir, file_name)
            )

    def _add_states(self, smoothing_window):
        """Add State objects to Pairs, Bonds, and Angles.
        Required step before optimization runs can begin.

        """
        try:
            self.smooth_rdfs
        except AttributeError:
            self.smooth_rdfs = False

        for pair in self.pairs:
            for state in self.states:
                pair._add_state(state, smoothing_window)

        for bond in self.bonds:
            for state in self.states:
                bond._add_state(state, smoothing_window)

        for angle in self.angles:
            for state in self.states:
                angle._add_state(state, smoothing_window)

        for dihedral in self.dihedrals:
            for state in self.states:
                dihedral._add_state(state, smoothing_window)

    def _update_potentials(self, iteration, smooth_pot, smoothing_window):
        """Update the potentials for the potentials to be optimized."""
        if self.optimization == "pairs":
            for pair in self.pairs:
                self._recompute_rdfs(pair, iteration, smoothing_window)
                pair._update_potential(smooth_pot, smoothing_window)
                save_table_potential(
                        pair.potential,
                        pair.r_range,
                        pair.dr,
                        iteration,
                        pair._potential_file
                )

        elif self.optimization == "bonds":
            for bond in self.bonds:
                self._recompute_distribution(bond, iteration, smoothing_window)
                bond._update_potential(smooth_pot, smoothing_window)
                save_table_potential(
                        bond.potential,
                        bond.l_range,
                        bond.dl,
                        iteration,
                        bond._potential_file
                )

        elif self.optimization == "angles":
            for angle in self.angles:
                print("Computing distribution from query simulation...")
                self._recompute_distribution(angle, iteration, smoothing_window)
                print("Updating the potential file...")
                angle._update_potential(smooth_pot, smoothing_window)
                print("Saving iteration potential file...")
                save_table_potential(
                        angle.potential,
                        angle.theta_range,
                        angle.dtheta,
                        iteration,
                        angle._potential_file
                )
                print(f"File saved to {angle._potential_file}")

        elif self.optimization == "dihedrals":
            for dihedral in self.dihedrals:
                self._recompute_distribution(dihedral, iteration, smoothing_window)
                dihedral._update_potential(smooth_pot, smoothing_window)
                save_table_potential(
                        dihedral.potential,
                        dihedral.phi_range,
                        dihedral.dphi,
                        iteration,
                        dihedral._potential_file
                )

    def _recompute_distribution(self, bond_object, iteration, smoothing_window):
        """Recompute the current distribution of bond lengths or angles"""
        for state in self.states:
            bond_object._compute_current_distribution(state, smoothing_window)
            bond_object._save_current_distribution(state, iteration=iteration)
            print("{0}, State: {1}, Iteration: {2}: {3:f}".format(
                    bond_object.name,
                    state.name,
                    iteration + 1,
                    bond_object._states[state]["f_fit"][iteration]
                )
            )

    def _recompute_rdfs(self, pair, iteration, smoothing_window):
        """Recompute the current RDFs for every state used for a given pair."""
        for state in self.states:
            pair._compute_current_rdf(state, smoothing_window)
            pair._save_current_rdf(state, iteration=iteration)
            print("Pair: {0}, State: {1}, Iteration: {2}: {3:f}".format(
                    pair.name,
                    state.name,
                    iteration + 1,
                    pair._states[state]["f_fit"][iteration]
                )
            )

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
            if pair.pair_type == "table" and self.optimization == "pairs":
                #TODO Fix handling of r_switch here?
                #pair.r_switch = pair.r_range[-5]
                potential_file = os.path.join(
                    self.potentials_dir, f"pair_pot.{pair.name}.txt"
                )
                pair.update_potential_file(potential_file)
                V = pair_tail_correction(
                        pair.r_range, pair.potential, pair.r_switch
                )
                pair.potential = V
                if self.optimization == "pairs":
                    iteration = 0
                else:
                    iteration = None
                save_table_potential(
                        pair.potential,
                        pair.r_range,
                        pair.dr,
                        iteration,
                        pair._potential_file
                )

        for bond in self.bonds:
            if bond.bond_type == "table" and bond._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"bond_pot.{bond.name}.txt"
                )
                bond.update_potential_file(potential_file)

                if self.optimization == "bonds":
                    iteration = 0
                else:
                    iteration = None

                save_table_potential(
                        bond.potential,
                        bond.l_range,
                        bond.dl,
                        iteration,
                        bond._potential_file
                )

        for angle in self.angles:
            if angle.angle_type == "table" and angle._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"angle_pot.{angle.name}.txt"
                )
                angle.update_potential_file(potential_file)
            elif angle.angle_type == "table" and angle._potential_file != "":
                potential_file = os.path.join(self.potentials_dir, f"angle_pot.{angle.name}")
                shutil.copyfile(angle._potential_file, potential_file)
                angle.update_potential_file(potential_file)

                if self.optimization == "angles":
                    iteration = 0
                else:
                    iteration = None

                save_table_potential(
                        angle.potential,
                        angle.theta_range,
                        angle.dtheta,
                        iteration,
                        angle._potential_file
                )

        for dihedral in self.dihedrals:
            if dihedral.dihedral_type == "table" and dihedral._potential_file == "":
                potential_file = os.path.join(
                        self.potentials_dir, f"dihedral_pot.{dihedral.name}.txt"
                )
                dihedral.update_potential_file(potential_file)

                if self.optimization == "dihedrals":
                    iteration = 0
                else:
                    iteration = None

                save_table_potential(
                        dihedral.potential,
                        dihedral.phi_range,
                        dihedral.dphi,
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

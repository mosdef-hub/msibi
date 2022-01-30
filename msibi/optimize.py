import os

import numpy as np

from msibi.potentials import tail_correction
from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import run_query_simulations


class MSIBI(object):
    """Management class for orchestrating an MSIBI optimization.

    Parameters
    ----------
    integrator : str, required 
        The integrator to use in the query simulation.
        See hoomd-blue.readthedocs.io/en/v2.9.6/module-md-integrate.html
    integrator_kwargs : dict, required 
        The args and their values required by the integrator chosen
    dt : float, required 
        The time step delta
    gsd_period : int, required 
        The number of frames between snapshots written to query.gsd
    n_iterations : int, required 
        Number of iterations.
    n_steps : int, required 
        How many steps to run the query simulations
    max_frames : int, required
        How many snapshots of the trajectories to use in calcualting
        relevant distributions (RDFs, bond distributions)
    potential_cutoff : float, required
        The upper range of the potential used for pair interactions
        If optimizing pair potentials, this value is also used
        as the r_max when calculating RDFs.
    r_min : float, default=1e-4
        The lower range of the potential used for pair interactions
        If optimizing pair potentials, this value is also used
        as the r_min when calculating RDFs.
    n_potential_points, int, default=101
        The number of bins to use in creating table pair potentials.
        If optiimizing pair potentials, this value is also used
        as the number of bins for RDF calculations.
    start_iteration : int, default 0
        Start optimization at start_iteration, useful for restarting.
    engine : str, default "hoomd"
        Engine that runs the simulations.
    verbose : bool, default False
        Whether to provide more information for debugging.

    Attributes
    ----------
    states : list of States
        All states to be used in the optimization procedure.
    pairs : list of Pairs
        All pairs to be used in the optimization procedure.
    bonds : list of Bonds
        All bonds to be used in the optimization procedure.
    angles : list of Angles
        All angles to be used in the optimization procedure.

    Methods
    -------
    add_state(state)
    add_bond(bond)
    add_angle(angle)
        Add the required interaction objects. See Pair.py and Bonds.py

    optimize_bonds(l_min, l_max)
        Calculates the target bond length distributions for each Bond
        in MSIBI.bonds and optimizes the bonding potential.

    optimize_angles(theta_min, theta_max)
        Calcualtes the target bond angle distribution for each Bond
        in MSIBI.angles and optimizes the angle potential.

    optimize_pairs(rdf_exclude_bonded, smooth_rdfs, r_switch)
        Calculates the target RDF for each Pair in MSIBI.pairs
        and optimizes the pair potential.

    """
    def __init__(
            self,
            integrator,
            integrator_kwargs,
            dt,
            gsd_period,
            n_steps,
            max_frames,
            potential_cutoff,
            r_min=1e-4,
            n_potential_points=101,
            verbose=False,
            engine="hoomd"
    ):
        if integrator == "hoomd.md.integrate.nve":
            raise ValueError("The NVE ensemble is not supported with MSIBI")
        
        self.pot_cutoff = potential_cutoff
        self.integrator = integrator
        self.integrator_kwargs = integrator_kwargs
        self.dt = dt
        self.gsd_period = gsd_period
        self.n_steps = n_steps
        self.max_frames = max_frames
        self.r_min = r_min
        self.n_pot_points = n_potential_points
        self.dr = self.pot_cutoff / (n_potential_points - 1)
        self.pot_r = np.arange(r_min, potential_cutoff + self.dr, self.dr)
        self.verbose = verbose
        self.engine = engine
        if engine == "hoomd":
            import hoomd
            self.HOOMD_VERSION = 2
        else:
            self.HOOMD_VERSION = None
        # Store all of the needed interaction objects
        self.states = []
        self.pairs = []
        self.bonds = []
        self.angles = []

    def add_state(self, state):
        state._opt = self
        self.states.append(state)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def add_bond(self, bond):
        self.bonds.append(bond)

    def add_angle(self, angle):
        self.angles.append(angle)

    def optimize_bonds(self, n_iterations, start_iteration=0):
        """Optimize the bond potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        start_iteration : int, default 0
            Start optimization at start_iteration, useful for restarting.

        """
        self.optimization = "bonds"
        self._add_states()
        self._initialize(potentials_dir=_dir)

        for n in range(start_iteration + n_iterations):
            print(f"-------- Iteration {n} --------")
            run_query_simulations(self.states, engine=self.engine)
            self._update_potentials(n)

    def optimize_angles(self, n_iterations, start_iteration=0):
        """Optimize the bond angle potentials

        Parameters
        ----------
        n_iterations : int, required 
            Number of iterations.
        start_iteration : int, default 0
            Start optimization at start_iteration, useful for restarting.

        """
        self.optimization = "angles"
        self._add_states()
        self._initialize(potentials_dir=_dir)

        for n in range(start_iteration + n_iterations):
            print(f"-------- Iteration {n} --------")
            run_query_simulations(self.states, engine=self.engine)
            self._update_potentials(n)

    def optimize_pairs(
        self,
        n_iterations,
        start_iteration=0,
        rdf_exclude_bonded=True,
        smooth_rdfs=True,
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
        rdf_exclude_bonded : bool, default=True
            Whether the RDF calculation should exclude correlations between bonded
            species.
        smooth_rdfs : bool, default=True
            Set to True to perform smoothing (Savitzky Golay) on the target
            and iterative RDFs.
        r_switch : float, optional, default=None
            The distance after which a tail correction is applied.
            If None, then self.pot_r[-5] is used.

        """
        self.optimization = "pairs"
        self.rdf_exclude_bonded = rdf_exclude_bonded
        self.smooth_rdfs = smooth_rdfs
        self.rdf_cutoff = self.pot_cutoff 
        self.rdf_r_range = np.array([self.r_min, self.rdf_cutoff + self.dr])
        self.rdf_n_bins = self.n_pot_points
        self.n_rdf_points = self.n_pot_points

        if r_switch is None:
            r_switch = self.pot_r[-5]
        self.r_switch = r_switch

        self._add_states()
        self._initialize(potentials_dir=_dir)

        for n in range(start_iteration + n_iterations):
            print(f"-------- Iteration {n} --------")
            run_query_simulations(self.states, engine=self.engine)
            self._update_potentials(n)

    def _add_states(self):
        """Add State objects to Pairs, Bonds, and Angles.
        Required step before optimization runs can begin.
        """
        for pair in self.pairs:
            for state in self.states:
                pair._add_state(state, smooth=self.smooth_rdfs)

        if self.bonds:
            for bond in self.bonds:
                for state in self.states:
                    bond._add_state(state)

        if self.angles:
            for angle in self.angles:
                for state in self.states:
                    angle._add_state(state)

        for state in self.states:
            state.HOOMD_VERSION = self.HOOMD_VERSION

    def _update_potentials(self, iteration):
        """Update the potentials for each object to be optimized."""
        if self.optimization == "pairs":
            for pair in self.pairs:
                self._recompute_rdfs(pair, iteration)
                pair._update_potential(self.pot_r, self.r_switch, self.verbose)
                pair._save_table_potential(self.pot_r, self.dr, iteration)

        elif self.optimization == "bonds":
            for bond in self.bonds:
                self._recompute_distribution(bond, iteration)
                bond._update_potential()

        elif self.optimization == "angles":
            for angle in self.angles:
                self._recompute_distribution(angle, iteration)
                angle._update_potential()

    def _recompute_distribution(self, bond_object, iteration):
        """Recompute the current distribution of bond lengths or angles"""
        for state in self.states:
            bond_object._compute_current_distribution(state)
            bond_object._save_current_distribution(state, iteration=iteration)

    def _recompute_rdfs(self, pair, iteration):
        """Recompute the current RDFs for every state used for a given pair."""
        for state in self.states:
            pair._compute_current_rdf(
                state,
                smooth=self.smooth_rdfs,
                verbose=self.verbose
            )
            pair._save_current_rdf(state, iteration=iteration, dr=self.dr)
            print(
                "pair {0}, state {1}, iteration {2}: {3:f}".format(
                    pair.name,
                    state.name,
                    iteration,
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

        table_potentials = []
        bonds = None
        angles = None

        for pair in self.pairs:
            potential_file = os.path.join(
                self.potentials_dir, f"pair_pot.{pair.name}.txt"
            )
            pair.potential_file = potential_file
            table_potentials.append((pair.type1, pair.type2, potential_file))

            V = tail_correction(self.pot_r, pair.potential, self.r_switch)
            pair.potential = V

            # This file is written for viewing of how the potential evolves.
            if self.optimization == "pairs":
                pair.save_table_potential(self.pot_r, self.dr, iteration=0)

            # This file is overwritten at each iteration and
            # used by Hoomd when performing query simulations
            pair.save_table_potential(self.pot_r, self.dr)

        if self.bonds:
            bonds = self.bonds
            for bond in self.bonds:
                if bond.bond_type == "quadratic":
                    bond.potential_file = os.path.join(
                            self.potentials_dir, f"bond_pot.{bond.name}.txt"
                    )

        if self.angles:
            angles = self.angles
                if angle.angle_type == "quadratic":
                    angle.potential_file = os.path.join(
                            self.potentials_dir, f"angle_pot.{angle.name}.txt"
                    )

        for state in self.states:
            state.save_runscript(
                n_steps=int(self.n_steps),
                integrator=self.integrator,
                integrator_kwargs=self.integrator_kwargs,
                dt=self.dt,
                gsd_period=self.gsd_period,
                table_potentials=table_potentials,
                table_width=len(self.pot_r),
                engine=self.engine,
                bonds=bonds,
                angles=angles
            )


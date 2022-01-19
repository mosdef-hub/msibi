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
    n_iterations : int, default 10
        Number of iterations.
    start_iteration : int, default 0
        Start optimization at start_iteration, useful for restarting.
    n_steps : int, default=1e6
        How many steps to run the query simulations
        The frequency to write trajectory information to query.gsd
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
    n_iterations : int
        The number of MSIBI iterations to perform.
    max_frames : int
        The maximum number of frames to include at once in RDF calculation
    rdf_cutoff : float
        The upper cutoff value for the RDF calculation.
    n_rdf_points : int
        The number of radius values used in the RDF calculation.
    dr : float
        The spacing of radius values.
    rdf_exclude_bonded : bool
        Whether the RDF calculation should exclude correlations between bonded
        species.
    pot_cutoff : float
        The upper cutoff value for the potential.
    pot_r : np.ndarray, shape=int((rdf_cutoff + dr) / dr)
        The radius values at which the potential is computed.
    r_switch : float
        The radius after which a tail correction is applied.
    """

    def __init__(
            self,
            integrator,
            integrator_kwargs,
            dt,
            gsd_period,
            n_iterations,
            n_steps,
            start_iteration=0,
            verbose=False,
            engine="hoomd"
    ):

        if integrator == "hoomd.md.integrate.nve":
            raise ValueError("The NVE ensemble is not supported with MSIBI")
        
        self.dt = dt
        self.integrator = integrator
        self.integrator_kwargs = integrator_kwargs
        self.gsd_period = gsd_period
        self.n_iterations = n_iterations
        self.n_steps = n_steps
        self.start_iteration = start_iteration
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

    def optimize_bonds(self):
        pass

    def optimize_angles(self):
        pass

    def optimize_pairs(
        self,
        max_frames,
        rdf_cutoff,
        potential_cutoff,
        r_min,
        r_switch,
        n_rdf_points,
        rdf_exclude_bonded,
        smooth_rdfs,
        _dir=None
    ):
        """Optimize the pair potentials

        Parameters
        ----------

        """
        # Set up attributes specific to pair potential optimization
        self.max_frames = max_frames
        self.rdf_cutoff = rdf_cutoff
        self.n_rdf_points = n_rdf_points
        self.dr = rdf_cutoff / (n_rdf_points - 1)
        self.r_min = r_min
        self.rdf_exclude_bonded rdf_exclude_bonded
        self.smooth_rdfs = smooth_rdfs
        self.rdf_r_range = np.array([self.r_min, self.rdf_cutoff + self.dr])
        self.rdf_n_bins = self.n_rdf_points
        # Sometimes the pot_cutoff and rdf_cutoff have different ranges,
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff
        self.pot_r = np.arange(self.r_min, self.pot_cutoff + self.dr, self.dr)
        if not r_switch:
            r_switch = self.pot_r[-5]
        self.r_switch = r_switch

        # Add all pair, angle, and bond objects
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

        self._initialize(
                engine=self.engine, 
                n_steps=int(self.n_steps),
                integrator=self.integrator,
                integrator_kwargs=self.integrator_kwargs,
                dt=self.dt,
                gsd_period=self.gsd_period,
                potentials_dir=_dir)

        for n in range(self.start_iteration + self.n_iterations):
            print(f"-------- Iteration {n} --------")
            run_query_simulations(self.states, engine=self.engine)
            self._update_potentials(n, engine)

    def _update_potentials(self, iteration, engine):
        """Update the potentials for each pair. """
        for pair in self.pairs:
            self._recompute_rdfs(pair, iteration)
            pair.update_potential(self.pot_r, self.r_switch, self.verbose)
            pair.save_table_potential(
                self.pot_r, self.dr, iteration=iteration, engine=engine
            )

    def _recompute_rdfs(self, pair, iteration):
        """Recompute the current RDFs for every state used for a given pair."""
        for state in self.states:
            pair.compute_current_rdf(
                state,
                smooth=self.smooth_rdfs,
                verbose=self.verbose
            )
            pair.save_current_rdf(state, iteration=iteration, dr=self.dr)
            print(
                "pair {0}, state {1}, iteration {2}: {3:f}".format(
                    pair.name,
                    state.name,
                    iteration,
                    pair._states[state]["f_fit"][iteration]
                )
            )

    def _initialize(
            self,
            engine,
            n_steps,
            integrator,
            integrator_kwargs,
            dt,
            gsd_period,
            potentials_dir
            ):
        """Create initial table potentials and the simulation input scripts.

        Parameters
        ----------
        engine : str, default 'hoomd'
            Engine used to run simulations
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

        if not os.path.isdir("rdfs"):
            os.mkdir("rdfs")

        table_potentials = []
        bonds = None
        angles = None

        for pair in self.pairs:
            potential_file = os.path.join(
                self.potentials_dir, f"pot.{pair.name}.txt"
            )
            pair.potential_file = potential_file
            table_potentials.append((pair.type1, pair.type2, potential_file))

            V = tail_correction(self.pot_r, pair.potential, self.r_switch)
            pair.potential = V

            # This file is written for viewing of how the potential evolves.
            pair.save_table_potential(
                self.pot_r, self.dr, iteration=0, engine=engine
            )

            # This file is overwritten at each iteration and actually used for
            # performing the query simulations.
            pair.save_table_potential(self.pot_r, self.dr, engine=engine)

        if self.bonds:
            bonds = self.bonds
        if self.angles:
            angles = self.angles

        for state in self.states:
            state.save_runscript(
                n_steps=n_steps,
                integrator=integrator,
                integrator_kwargs=integrator_kwargs,
                dt=dt,
                gsd_period=gsd_period,
                table_potentials=table_potentials,
                table_width=len(self.pot_r),
                engine=engine,
                bonds=bonds,
                angles=angles
            )

import os

import numpy as np

from msibi.potentials import tail_correction
from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import run_query_simulations


class MSIBI(object):
    """Management class for orchestrating an MSIBI optimization.

    Parameters
    ----------
    rdf_cutoff : float
        The upper cutoff value for the RDF calculation.
    n_rdf_points : int
        The number of radius values used in RDF calculations.
    max_frames : int, default 10
        The maximum number of frames to include at once in RDF calculation
        The RDF calculation will accumulate an average RDF over the range
        [-max_frames:-1] of the trajectory. This is also how a target_rdf
        will be calculated in Pair.add_state
    pot_cutoff : float, default None
        The upper cutoff value for the potential. If None is provided,
        rdf_cutoff is used.
    r_switch : float, default None
        The radius after which a tail correction is applied. If None is
        provided, pot_r[-5] is used.
    rdf_exclude_bonded : bool, default False
        Whether the RDF calculation should exclude correlations between bonded
        species.
    smooth_rdfs : bool, default False
        Whether to use a smoothing function to reduce the noise in the RDF data.
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
        rdf_cutoff,
        n_rdf_points,
        max_frames=10,
        pot_cutoff=None,
        r_switch=None,
        rdf_exclude_bonded=False,
        smooth_rdfs=False,
        verbose=False
    ):

        rmin = 1e-4
        self.verbose = verbose
        self.states = []
        self.pairs = []
        self.bonds = []
        self.angles = []
        self.n_iterations = 10  # Can be overridden in optimize().
        self.max_frames = max_frames

        self.rdf_cutoff = rdf_cutoff
        self.n_rdf_points = n_rdf_points
        self.dr = rdf_cutoff / (n_rdf_points - 1)
        self.rdf_exclude_bonded = rdf_exclude_bonded
        self.smooth_rdfs = smooth_rdfs
        self.rdf_r_range = np.array([rmin, self.rdf_cutoff + self.dr])
        self.rdf_n_bins = self.n_rdf_points

        # Sometimes the pot_cutoff and rdf_cutoff have different ranges,
        # e.g. to look at long-range correlations
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff

        self.pot_r = np.arange(rmin, self.pot_cutoff + self.dr, self.dr)

        if not r_switch:
            r_switch = self.pot_r[-5]
        self.r_switch = r_switch

    def add_state(self, state):
        state._opt = self
        self.states.append(state)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def add_bond(self, bond):
        self.bonds.append(bond)

    def add_angle(self, angle):
        self.angles.append(angle)


    def optimize(
        self,
        integrator,
        integrator_kwargs,
        dt,
        gsd_period,
        n_iterations=10,
        start_iteration=0,
        n_steps=1e6,
        engine="hoomd",
        _dir=None
    ):
        """Optimize the pair potentials

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

        References
        ----------
        Please cite the following paper:

        .. [1] T.C. Moore et al., "Derivation of coarse-grained potentials via
           multistate iterative Boltzmann inversion," Journal of Chemical
           Physics, vol. 140, pp. 224104, 2014.
        """
        if integrator == "hoomd.md.integrate.nve":
            raise ValueError("The NVE ensemble is not supported with MSIBI")

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

        if engine == "hoomd":
            import hoomd
            HOOMD_VERSION = 2
        else:
            HOOMD_VERSION = None

        if self.verbose:
            print(f"Using HOOMD version {HOOMD_VERSION}.")

        for state in self.states:
            state.HOOMD_VERSION = HOOMD_VERSION

        self.n_iterations = n_iterations
        self._initialize(
                engine=engine, 
                n_steps=int(n_steps),
                integrator=integrator,
                integrator_kwargs=integrator_kwargs,
                dt=dt,
                gsd_period=gsd_period,
                potentials_dir=_dir)

        for n in range(start_iteration + self.n_iterations):
            print(f"-------- Iteration {n} --------")
            run_query_simulations(self.states, engine=engine)
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

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
    n_points : int
        The number of radius values used in RDF calculations.
    max_frames : int
        The maximum number of frames to include at once in RDF calculation
        The RDF calculation will accumulate an average RDF over the range
        [-max_frames:-1] of the trajectory. This is also how a target_rdf
        will be calculated in Pair.add_state
    pot_cutoff : float, optional, default=rdf_cutoff
        The upper cutoff value for the potential.
    r_switch : float, optional, default=pot_r[-5]
        The radius after which a tail correction is applied.
    smooth_rdfs : bool, optional, default=False
        Use a smoothing function to reduce the noise in the RDF data.
    verbose : bool
        Whether to provide more information for debugging (default False)

    Attributes
    ----------
    states : list of States
        All states to be used in the optimization procedure.
    pairs : list of Pairs
        All pairs to be used in the optimization procedure.
    n_iterations : int, optional, default=10
        The number of MSIBI iterations to perform.
    rdf_cutoff : float
        The upper cutoff value for the RDF calculation.
    n_rdf_points : int
        The number of radius values used in the RDF calculation.
    dr : float, default=rdf_cutoff / (n_points - 1)
        The spacing of radius values.
    pot_cutoff : float, optional, default=rdf_cutoff
        The upper cutoff value for the potential.
    pot_r : np.ndarray, shape=(int((rdf_cutoff + dr) / dr),)
        The radius values at which the potential is computed.
    r_switch : float, optional, default=pot_r[-1] - 5 * dr
        The radius after which a tail correction is applied.
    """

    def __init__(
        self,
        rdf_cutoff,
        n_rdf_points,
        max_frames=1000,
        pot_cutoff=None,
        r_switch=None,
        smooth_rdfs=False,
        verbose=False
    ):

        self.verbose = verbose
        self.states = []
        self.pairs = []
        self.n_iterations = 10  # Can be overridden in optimize().
        self.max_frames = max_frames

        self.rdf_cutoff = rdf_cutoff
        self.n_rdf_points = n_rdf_points
        self.dr = rdf_cutoff / (n_rdf_points - 1)
        self.smooth_rdfs = smooth_rdfs
        self.rdf_r_range = np.array([0.0, self.rdf_cutoff + self.dr])
        self.rdf_n_bins = self.n_rdf_points

        # Sometimes the pot_cutoff and rdf_cutoff have different ranges,
        # e.g. to look at long-range correlations
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff

        self.pot_r = np.arange(0.0, self.pot_cutoff + self.dr, self.dr)

        if not r_switch:
            r_switch = self.pot_r[-5]
        self.r_switch = r_switch

    def add_state(self, state):
        state._opt = self
        self.states.append(state)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def optimize(
        self,
        n_iterations=10,
        start_iteration=0,
        engine="hoomd",
    ):
        """Optimize the pair potentials

        Parameters
        ----------
        n_iterations : int
            Number of iterations. (default 10)
        start_iteration : int
            Start optimization at start_iteration, useful for restarting.
            (default 0)
        engine : str
            Engine that runs the simulations. (default "hoomd")

        References
        ----------
        Please cite the following paper:

        .. [1] T.C. Moore et al., "Derivation of coarse-grained potentials via
           multistate iterative Boltzmann inversion," Journal of Chemical
           Physics, vol. 140, pp. 224104, 2014.
        """
        for pair in self.pairs:
            for state in self.states:
                pair._add_state(state)

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
        self._initialize(engine=engine)

        for n in range(start_iteration + self.n_iterations):
            print("-------- Iteration {n} --------".format(**locals()))
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

    def _initialize(self, engine="hoomd", potentials_dir=None):
        """Create initial table potentials and the simulation input scripts.

        Parameters
        ----------
        engine : str, optional, default='hoomd'
            Engine used to run simulations
        potentials_dir : path, optional, default="'working_dir'/potentials"
            Directory to store potential files
        """
        if not potentials_dir:
            self.potentials_dir = os.path.join(os.getcwd(), "potentials")
        else:
            self.potentials_dir = potentials_dir

        if not os.path.isdir(self.potentials_dir):
            os.mkdir(self.potentials_dir)

        if not os.path.isdir("rdfs"):
            os.mkdir("rdfs")

        table_potentials = []
        for pair in self.pairs:
            potential_file = os.path.join(
                self.potentials_dir, "pot.{0}.txt".format(pair.name)
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

        for state in self.states:
            state.save_runscript(
                table_potentials, table_width=len(self.pot_r), engine=engine
            )

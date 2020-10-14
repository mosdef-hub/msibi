##############################################################################
# MSIBI: A package for optimizing coarse-grained force fields using multistate
#   iterative Boltzmann inversion.
# Copyright (c) 2017 Vanderbilt University and the Authors
#
# Authors: Christoph Klein, Timothy C. Moore
# Contributors: Davy Yue
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files, to deal
# in MSIBI without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of MSIBI, and to permit persons to whom MSIBI is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of MSIBI.
#
# MSIBI IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH MSIBI OR THE USE OR OTHER DEALINGS ALONG WITH
# MSIBI.
#
# You should have received a copy of the MIT license.
# If not, see <https://opensource.org/licenses/MIT/>.
##############################################################################


from __future__ import division

import logging
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
        The number of radius values.
    pot_cutoff : float, optional, default=rdf_cutoff
        The upper cutoff value for the potential.
    r_switch : float, optional, default=pot_r[-5]
        The radius after which a tail correction is applied.
    smooth_rdfs : bool, optional, default=False
        Use a smoothing function to reduce the noise in the RDF data.
    max_frames : int
        The maximum number of frames to include at once in RDF calculation

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
        pot_cutoff=None,
        r_switch=None,
        smooth_rdfs=False,
        max_frames=1e3,
    ):

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

    def optimize(
        self, states, pairs, n_iterations=10, engine="hoomd", start_iteration=0
    ):
        """Optimize the pair potentials

        Parameters
        ----------
        states : array_like, len=n_states, dtype=msibi.State
            List of states used to optimize pair potentials.
        pairs : array_like, len=n_pairs, dtype=msibi.Pair
            List of pairs being optimized.
        n_iterations : int, optional
            Number of iterations.
        engine : str, optional
            Engine that runs the simulations.
        start_iteration : int, optional
            Start optimization at start_iteration, useful for restarting.

        References
        ----------
        Please cite the following paper:

        .. [1] T.C. Moore et al., "Derivation of coarse-grained potentials via
           multistate iterative Boltzmann inversion," Journal of Chemical
           Physics, vol. 140, pp. 224104, 2014.

        """

        if engine == "hoomd":
            try:
                import hoomd

                HOOMD_VERSION = 2
            except ImportError:
                try:
                    import hoomd_script

                    HOOMD_VERSION = 1
                except ImportError:
                    raise ImportError("Cannot import hoomd")
        else:  # don't need a hoomd version if not using hoomd
            HOOMD_VERSION = None

        for pair in pairs:
            for state, data in pair.states.items():
                if len(data["target_rdf"]) != self.n_rdf_points:
                    raise ValueError(
                        "Target RDF in {} of pair {} is not the "
                        "same length as n_rdf_points.".format(state.name, pair.name)
                    )

        for state in states:
            state.HOOMD_VERSION = HOOMD_VERSION

        self.states = states
        self.pairs = pairs
        self.n_iterations = n_iterations
        self.initialize(engine=engine)

        for n in range(start_iteration + self.n_iterations):
            logging.info("-------- Iteration {n} --------".format(**locals()))
            run_query_simulations(self.states, engine=engine)
            self._update_potentials(n, engine)

    def _update_potentials(self, iteration, engine):
        """Update the potentials for each pair. """
        for pair in self.pairs:
            self._recompute_rdfs(pair, iteration)
            pair.update_potential(self.pot_r, self.r_switch)
            pair.save_table_potential(
                self.pot_r, self.dr, iteration=iteration, engine=engine
            )

    def _recompute_rdfs(self, pair, iteration):
        """Recompute the current RDFs for every state used for a given pair. """
        for state in pair.states:
            pair.compute_current_rdf(
                state,
                self.rdf_r_range,
                n_bins=self.rdf_n_bins,
                smooth=self.smooth_rdfs,
                max_frames=self.max_frames,
            )
            pair.save_current_rdf(state, iteration=iteration, dr=self.dr)
            logging.info(
                "pair {0}, state {1}, iteration {2}: {3:f}".format(
                    pair.name,
                    state.name,
                    iteration,
                    pair.states[state]["f_fit"][iteration],
                )
            )

    def initialize(self, engine="hoomd", potentials_dir=None):
        """
        Create initial table potentials and the simulation input scripts.

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
            pair.save_table_potential(self.pot_r, self.dr, iteration=0, engine=engine)

            # This file is overwritten at each iteration and actually used for
            # performing the query simulations.
            pair.save_table_potential(self.pot_r, self.dr, engine=engine)

        for state in self.states:
            state.save_runscript(
                table_potentials, table_width=len(self.pot_r), engine=engine
            )

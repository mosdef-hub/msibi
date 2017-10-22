##############################################################################
# MSIBI: A package for optimizing coarse-grained force fields using multistate
#   iterative Boltzmann inversion.
# Copyright (c) 2017 Vanderbilt University and the Authors
#
# Authors: Christoph Klein, Timothy C. Moore
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
import os

import mdtraj as md

import numpy as np
from six import string_types

from msibi.utils.exceptions import UnsupportedEngine
from msibi.utils.error_calculation import calc_similarity
from msibi.potentials import tail_correction, head_correction, alpha_array
from msibi.utils.smoothing import savitzky_golay
from msibi.utils.find_exclusions import find_1_n_exclusions


class Pair(object):
    """A pair interaction to be optimized.

    Attributes
    ----------
    name : str
        Pair name.
    pairs : array-like, shape=(n_pairs, 2), dtype=int, optional, default=None
        Each row gives the indices of two atoms representing a pair.
    potential : func
        Values of the potential at every pot_r.

    """
    def __init__(self, type1, type2, potential, head_correction_form='linear'):
        self.type1 = str(type1)
        self.type2 = str(type2)
        self.name = '{0}-{1}'.format(self.type1, self.type2)
        self.potential_file = ''
        self.states = dict()
        if isinstance(potential, string_types):
            self.potential = np.loadtxt(potential)[:, 1]
            # TODO: this could be dangerous
        else:
            self.potential = potential
        self.previous_potential = None
        self.head_correction_form = head_correction_form

    def add_state(self, state, target_rdf, alpha, pair_indices=None,
                  alpha_form='linear'):
        """Add a state to be used in optimizing this pair.

        Parameters
        ----------
        state : State
            A state object.
        target_rdf : np.ndarray, shape=(n_bins, 2), dtype=float
            Coarse-grained target RDF.
        alpha : float
            The alpha value used to scale the weight of this state.
        pair_indices : array-like, shape=(n_pairs, 2), dtype=int, optional, default=None
            Each row gives the indices of two atoms representing a pair.
        alpha_form : str
            For alpha as a function of r, gives form of alpha function
        """
        self.states[state] = {'target_rdf': target_rdf,
                              'current_rdf': None,
                              'alpha': alpha,
                              'alpha_form': alpha_form,
                              'pair_indices': pair_indices,
                              'f_fit': []}

    def select_pairs(self, state, exclude_up_to=0):
        """Select pairs based on a topology and exclusions.

        Parameters
        ----------
        state : State
            A state object, contains a topology from which to select pairs
        exclude_up_to : int
            Exclude pairs separated by exclude_up_to or fewer bonds, default=0
        """
        if state.top_path:
            top = md.load(state.top_path).topology
        else:
            top = md.load(state.traj_path).topology
        pairs = top.select_pairs("name '{0}'".format(self.type1),
                                 "name '{0}'".format(self.type2))
        if exclude_up_to is not None:
            to_delete = find_1_n_exclusions(top, pairs, exclude_up_to)
            pairs = np.delete(pairs, to_delete, axis=0)
        self.states[state]['pair_indices'] = pairs

    def compute_current_rdf(self, state, r_range, n_bins, smooth=True,
                            max_frames=1e3):
        """ """
        pairs = self.states[state]['pair_indices']
        # TODO: More elegant way to handle units.
        #       See https://github.com/ctk3b/msibi/issues/2
        g_r_all = None
        first_frame = 0
        max_frames = int(max_frames)
        for last_frame in range(max_frames,
                                state.traj.n_frames + max_frames,
                                max_frames):
            r, g_r = md.compute_rdf(state.traj[first_frame:last_frame],
                                    pairs, r_range=r_range / 10, n_bins=n_bins)
            if g_r_all is None:
                g_r_all = np.zeros_like(g_r)
            g_r_all += g_r * len(state.traj[first_frame:last_frame]) / state.traj.n_frames
            first_frame = last_frame
        r *= 10
        rdf = np.vstack((r, g_r_all)).T
        self.states[state]['current_rdf'] = rdf

        if smooth:
            current_rdf = self.states[state]['current_rdf']
            current_rdf[:, 1] = savitzky_golay(current_rdf[:, 1], 9, 2, deriv=0, rate=1)
            for row in current_rdf:
                row[1] = np.maximum(row[1], 0)

        # Compute fitness function comparing the two RDFs.
        f_fit = calc_similarity(rdf[:, 1], self.states[state]['target_rdf'][:, 1])
        self.states[state]['f_fit'].append(f_fit)

    def save_current_rdf(self, state, iteration, dr):
        """Save the current rdf

        Parameters
        ----------
        state : State
            A state object
        iteration : int
            Current iteration step, used in the filename
        dr : float
            The RDF bin size
        """
        if not os.path.isdir('./rdfs'):
            os.makedirs('./rdfs')
        filename = './rdfs/pair_{0}-state_{1}-step{2}.txt'.format(
                self.name, state.name, iteration)
        rdf = self.states[state]['current_rdf']
        rdf[:, 0] -= dr / 2
        np.savetxt(filename, rdf)

    def update_potential(self, pot_r, r_switch=None):
        """Update the potential using all states. """
        self.previous_potential = np.copy(self.potential)
        for state in self.states:
            kT = state.kT
            alpha0 = self.states[state]['alpha']
            form = self.states[state]['alpha_form']
            alpha = alpha_array(alpha0, pot_r, form=form)

            current_rdf = self.states[state]['current_rdf'][:, 1]
            target_rdf = self.states[state]['target_rdf'][:, 1]

            # For cases where rdf_cutoff != pot_cutoff, only update the
            # potential using RDF values < pot_cutoff.
            unused_rdf_vals = current_rdf.shape[0] - self.potential.shape[0]
            if unused_rdf_vals != 0:
                current_rdf = current_rdf[:-unused_rdf_vals]
                target_rdf = target_rdf[:-unused_rdf_vals]

            # The actual IBI step.
            self.potential += (kT * alpha * np.log(current_rdf / target_rdf) /
                               len(self.states))

        # Apply corrections to ensure continuous, well-behaved potentials.
        self.potential = tail_correction(pot_r, self.potential, r_switch)
        self.potential = head_correction(pot_r, self.potential,
                self.previous_potential, self.head_correction_form)

    def save_table_potential(self, r, dr, iteration=0, engine='hoomd'):
        """Save the table potential to a file usable by the MD engine. """
        V = self.potential
        F = -1.0 * np.gradient(V, dr)
        data = np.vstack([r, V, F])

        basename = os.path.basename(self.potential_file)
        basename = 'step{0:d}.{1}'.format(iteration, basename)
        dirname = os.path.dirname(self.potential_file)
        iteration_filename = os.path.join(dirname, basename)

        # TODO: Factor out for separate engines.
        if engine.lower() == 'hoomd':
            # This file is overwritten at each iteration and actually used for
            # performing the query simulations.
            np.savetxt(self.potential_file, data.T)
            # This file is written for viewing of how the potential evolves.
            np.savetxt(iteration_filename, data.T)
        else:
            raise UnsupportedEngine(engine)

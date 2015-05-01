import os

import mdtraj as md
import numpy as np
from six import string_types

from msibi.utils.exceptions import UnsupportedEngine
from msibi.utils.error_calculation import calc_similarity
from msibi.potentials import tail_correction, head_correction, alpha_array
from msibi.utils.smoothing import savitzky_golay


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
    def __init__(self, type1, type2, potential):
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

    def add_state(self, state, target_rdf, alpha, pair_indices,
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

        """
        self.states[state] = {'target_rdf': target_rdf,
                              'current_rdf': None,
                              'alpha': alpha,
                              'alpha_form': alpha_form,
                              'pair_indices': pair_indices,
                              'f_fit': []}

    def compute_current_rdf(self, state, r_range, n_bins, smooth=True):
        """ """
        pairs = self.states[state]['pair_indices']
        # TODO: More elegant way to handle units.
        #       See https://github.com/ctk3b/msibi/issues/2
        r, g_r = md.compute_rdf(state.traj, pairs, r_range=r_range / 10,
                                n_bins=n_bins)
        r *= 10
        rdf = np.vstack((r, g_r)).T
        self.states[state]['current_rdf'] = rdf

        # Compute fitness function comparing the two RDFs.
        f_fit = calc_similarity(rdf[:, 1], self.states[state]['target_rdf'][:, 1])
        self.states[state]['f_fit'].append(f_fit)

        if smooth:
            self.states[state]['current_rdf'][:, 1] = savitzky_golay(
                self.states[state]['current_rdf'][:, 1], 5, 1, deriv=0, rate=1)

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
        self.potential = head_correction(pot_r, self.potential, self.previous_potential)

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

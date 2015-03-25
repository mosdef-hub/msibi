import os

import mdtraj as md
import numpy as np

from six import string_types
from msibi.utils.exceptions import UnsupportedEngine
from msibi.potentials import tail_correction, head_correction, calc_alpha_array


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
            #TODO: this could be dangerous
        else:
            self.potential = potential

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

    def compute_current_rdf(self, state, r_range, dr):
        """ """
        pairs = self.states[state]['pair_indices']
        # TODO: fix units
        r, g_r = md.compute_rdf(state.traj, pairs, r_range=r_range / 10,
                bin_width=dr / 10)
        r *= 10
        rdf = np.vstack((r, g_r)).T
        self.states[state]['current_rdf'] = rdf


    def save_current_rdf(self, state, iteration):
        filename = 'rdfs/pair_{0}-state_{1}-step{2}.txt'.format(
                self.name, state.name, iteration)
        rdf = self.states[state]['current_rdf']
        np.savetxt(filename, rdf)


    def update_potential(self, pot_r, r_switch=None):
        """ """
        self.previous_potential = np.copy(self.potential)
        for state in self.states:
            kT = state.kT
            alpha0 = self.states[state]['alpha']
            form = self.states[state]['alpha_form']
            alpha = calc_alpha_array(alpha0, pot_r, form=form)

            current_rdf = self.states[state]['current_rdf'][:, 1]
            target_rdf = self.states[state]['target_rdf'][:, 1]
            f_fit = calc_similarity(current_rdf, target_rdf)
            pair.states[state]['f_fit'].append(f_fit)
            unused_rdf_vals = current_rdf.shape[0] - self.potential.shape[0]
            if unused_rdf_vals != 0:
                current_rdf = current_rdf[:-unused_rdf_vals]
                target_rdf = target_rdf[:-unused_rdf_vals]

            self.potential += (kT * alpha * np.log(current_rdf / target_rdf) /
                len(self.states))

        V = tail_correction(pot_r, self.potential, r_switch)
        V = head_correction(pot_r, self.potential, self.previous_potential)
        self.potential = V

    def save_table_potential(self, r, dr, iteration=None, engine='hoomd'):
        """ """
        V = self.potential
        F = -1.0 * np.gradient(V, dr)
        data = np.vstack([r, V, F])

        if iteration is not None:
            assert isinstance(iteration, int)
            basename = os.path.basename(self.potential_file)
            basename = 'step{0:d}.{1}'.format(iteration, basename)
            dirname = os.path.dirname(self.potential_file)
            iteration_filename = os.path.join(dirname, basename)

        if engine.lower() == 'hoomd':
            np.savetxt(self.potential_file, data.T)
            if iteration is not None:
                np.savetxt(iteration_filename, data.T)
        else:
            raise UnsupportedEngine(engine)

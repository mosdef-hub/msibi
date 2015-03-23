import os

import mdtraj as md
import numpy as np

from msibi.optimize import R_RANGE, R, DR


class Pair(object):
    """A pair interaction to be optimized.

    Attributes
    ----------
    name : str
        Pair name.
    pairs : array-like, shape=(n_pairs, 2), dtype=int, optional, default=None
        Each row gives the indices of two atoms representing a pair.

    potential : func, optional, default=lennard_jones_12_6
        Form of the potential function.

    """
    def __init__(self, type1, type2, potential):
        self.type1 = str(type1)
        self.type2 = str(type2)
        self.name = '{0}-{1}'.format(self.type1, self.type2)
        self.potential = potential
        self.states = dict()

    def add_state(self, state, target_rdf, alpha, pair_indices):
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
                              'pair_indices': pair_indices}

    def compute_current_rdf(self, state, save_txt=False):
        """ """
        pairs = self.states[state]['pair_indices']
        r, g_r = md.compute_rdf(state.traj, pairs, r_range=R_RANGE * 10 , bin_width=DR * 10)
        rdf = np.vstack((r, g_r)).T
        self.states[state]['current_rdf'] = rdf

        if save_txt:
            filename = 'pair_{0}-state_{1}.txt'.format(self.name, state.name)
            np.savetxt(filename, rdf)

    def update_potential(self):
        """ """
        for state in self.states:
            alpha = self.states[state]['alpha']
            kT = state.kT
            current_rdf = self.states[state]['current_rdf'][1]
            target_rdf = self.states[state]['target_rdf'][1]

            self.potential += kT * alpha * np.log(current_rdf / target_rdf)

    def save_table_potential(self, filename, iteration=None):
        """ """
        V = self.potential
        F = -1.0 * np.gradient(V, DR)
        data = np.vstack([R, V, F])

        if iteration:
            assert isinstance(iteration, int)
            basename = os.path.basename(filename)
            basename = 'step{0:d}.{1}'.format(iteration, basename)
            dirname = os.path.dirname(filename)
            filename = os.path.join(dirname, basename)
        np.savetxt(filename, data.T)

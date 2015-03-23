import mdtraj as md
import numpy as np

from msibi.msibi import R_RANGE


class Pair(object):
    """A pair interaction to be optimized.

    Attributes
    ----------
    name : str
        Pair name.
    pairs : array-like, shape=(n_pairs, 2), dtype=int, optional, default=None
        Each row gives the indices of two atoms representing a pair.
    target_rdf : np.ndarray, shape=(n_bins, 2), dtype=float
        Coarse-grained target RDF.
    current_rdf : np.ndarray, shape=(n_bins, 2), dtype=float
        Coarse-grained RDF at the current iteration.
    potential : func, optional, default=lennard_jones_12_6
        Form of the potential function.

    """
    def __init__(self, name, potential):
        self.name = name
        self.potential = potential
        self.states = dict()

    def add_state(self, state, target_rdf, alpha, pair_indices):
        """Add a state to be used in optimizing this pair.

        state : State
        target_rdf : np.ndarray, shape=(n, 2), dtype=float
        alpha : float
        pair_indices :

        """
        self.states[state] = {'target_rdf': target_rdf,
                              'current_rdf': None,
                              'alpha': alpha,
                              'pair_indices': pair_indices}

    def compute_current_rdf(self, state, save_txt=False):
        """ """
        pairs = self.states[state]['pair_indices']
        r, g_r = md.compute_rdf(state.traj, pairs, r_range=R_RANGE)
        rdf = np.vstack((r, g_r))
        self.states[state]['current_rdf'] = rdf

        if save_txt:
            filename = 'pair_{0}-state_{1}.txt'.format(self.name, state.name)
            np.savetxt(filename, rdf)

    def update_potential(self):
        """ """
        for state in self.states:
            alpha = self.states[state]['alpha']
            kT = state.kT
            current_rdf = self.states[state]['current_rdf']
            target_rdf = self.states[state]['target_rdf']

            self.potential += kT * alpha * np.log(current_rdf / target_rdf)

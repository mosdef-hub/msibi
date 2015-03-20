import mdtraj as md

from msibi.potentials import lennard_jones_12_6


class Pair(object):
    """A pair interaction to be optimized.

    Attributes
    ----------

    states : array-like, shape=(n_states,), dtype=object
        All states that should be used to optimize this pair.
    alphas : array-like, shape=(n_states),), dtype=float
        The alpha values used to scale the influence of each state. These
        values correspond to the states in self.states.
    pairs : array-like, shape=(n_pairs, 2), dtype=int, optional, default=None
        Each row gives the indices of two atoms representing a pair.
    potential : func, optional, default=lennard_jones_12_6


    """
    def __init__(self, states, alphas, atom_indices, potential=lennard_jones_12_6):
        assert len(states) == len(alphas), 'Must provide one alpha value per state.'

        self.states = self.prepare_states(states)
        self.alphas = alphas
        self.atom_indices = atom_indices
        self.potential = potential

    def prepare_states(self, states):
        """Compute the target RDF's for every state considered for this pair. """
        states = dict.fromkeys(states)
        for state in states:
            target_rdf = md.compute_rdf(state.traj, pairs=self.pairs, r_range=[0.0, 2.0])
            states[state] = {'target_rdf': target_rdf, 'current_rdf': None}
        return states
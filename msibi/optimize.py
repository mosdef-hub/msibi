import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('white', {'legend.frameon': True,
                        'axes.edgecolor': '0.0',
                        'axes.linewidth': 1.0,
                        'xtick.direction': 'in',
                        'ytick.direction': 'in',
                        'xtick.major.size': 4.0,
                        'ytick.major.size': 4.0})


from msibi.potentials import tail_correction
from msibi.workers import run_query_simulations


class MSIBI(object):
    """
    """

    def __init__(self, rdf_cutoff, dr, pot_cutoff=None, r_switch=None):
        self.states = []
        self.pairs = []
        self.n_iterations = 10
        self.rdf_cutoff = rdf_cutoff
        self.dr = dr

        # TODO: description of use for pot vs rdf cutoff
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff
        # TODO: note on why the potential needs to be shortened to match the RDF
        self.pot_r = np.arange(0.0, pot_cutoff + dr, dr)

        if not r_switch:
            r_switch = self.pot_r[-1] - 5 * dr
        self.r_switch = r_switch

    def optimize(self, states, pairs, n_iterations=10, engine='hoomd'):
        """
        """
        self.states = states
        self.pairs = pairs
        self.n_iterations = n_iterations
        self.initialize(engine=engine)
        for n in range(self.n_iterations):
            run_query_simulations(self.states, engine=engine)

            for pair in self.pairs:
                for state in pair.states:
                    pair.compute_current_rdf(state,
                            np.array([0.0, self.rdf_cutoff + 2 * self.dr]), 
                            self.dr)
                    pair.save_current_rdf(state, iteration=n)
                pair.update_potential(self.pot_r, self.r_switch)
                pair.save_table_potential(self.pot_r, self.dr, iteration=n, 
                        engine=engine)
            print("Finished iteration {0}".format(n))

    def initialize(self, engine='hoomd', potentials_dir=None):
        """Create initial table potentials and the simulation input scripts.

        Parameters
        ----------
        engine : str, optional, default='hoomd'
        potentials_dir : path, optional, default="current_working_dir/potentials"

        """
        if not potentials_dir:
            self.potentials_dir = os.path.join(os.getcwd(), 'potentials')
        else:
            self.potentials_dir = potentials_dir
        try:
            os.mkdir(self.potentials_dir)
        except OSError:
            # TODO: warning and maybe a "make backups" feature
            pass

        table_potentials = []
        for pair in self.pairs:
            potential_file = os.path.join(self.potentials_dir,
                                          'pot.{0}.txt'.format(pair.name))
            pair.potential_file = potential_file

            table_potentials.append((pair.type1, pair.type2, potential_file))

            V = tail_correction(self.pot_r, pair.potential, self.r_switch)
            pair.potential = V
            # This file is written for viewing of how the potential evolves.
            pair.save_table_potential(self.pot_r, self.dr, iteration=0,
                                      engine=engine)
            # This file is overwritten at each iteration and actually used for
            # performing the query simulations.
            pair.save_table_potential(self.pot_r, self.dr, engine=engine)

        for state in self.states:
            state.save_runscript(table_potentials, table_width=len(self.pot_r),
                                 engine=engine)

    def plot(self):
        """Generate plots showing the evolution of each pair potential. """
        sns.set_palette(
            sns.cubehelix_palette(self.n_iterations, start=.5, rot=-.75))
        try:
            os.mkdir('figures')
        except OSError:
            pass
        for pair in self.pairs:
            for n in range(self.n_iterations):
                potential_file = os.path.join(self.potentials_dir, 'step{0:d}.{1}'.format(
                        n, os.path.basename(pair.potential_file)))
                data = np.loadtxt(potential_file)
                plt.plot(data[:, 0], data[:, 1],
                         linewidth=1, label='n={0:d}'.format(n))
            plt.xlabel('r')
            plt.ylabel('V(r)')
            plt.legend()
            plt.savefig('figures/{0}.pdf'.format(pair.name))

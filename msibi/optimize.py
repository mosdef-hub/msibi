import multiprocessing as mp
from multiprocessing.dummy import Pool
import os
from subprocess import Popen

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
sns.color_palette("GnBu_d")

from msibi.utils.exceptions import UnsupportedEngine


class MSIBI(object):
    """
    """

    def __init__(self, rdf_cutoff, dr, pot_cutoff=None):
        self.states = None
        self.pairs = None
        self.n_iterations = 10
        self.rdf_cutoff = rdf_cutoff
        self.dr = dr
        self.rdf_r = np.arange(0.0, rdf_cutoff, dr)

        # TODO: description of use for pot vs rdf cutoff
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff
        # TODO: note on why the potential needs to be shortened to match the RDF
        self.pot_r = np.arange(0.0, pot_cutoff - dr, dr)

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
                    pair.compute_current_rdf(state, np.array([0.0, self.rdf_cutoff]), self.dr)
                pair.update_potential()
                pair.save_table_potential(self.pot_r, self.dr, iteration=n, engine=engine)
            print("Finished iteration {0}".format(n))

    def initialize(self, engine='hoomd', potentials_dir=None):
        """

        Parameters
        ----------
        states : list of States
        pairs : list of Pairs
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
            potential_file = os.path.join(self.potentials_dir, 'pot.{0}.txt'.format(pair.name))
            pair.potential_file = potential_file

            table_potentials.append((pair.type1, pair.type2, potential_file))

            # This file is written for later viewing of how the potential evolves.
            pair.save_table_potential(self.pot_r, self.dr, iteration=0, engine=engine)
            # This file is overwritten at each iteration and actually used for the
            # simulation.
            pair.save_table_potential(self.pot_r, self.dr, engine=engine)

        for state in self.states:
            state.save_runscript(table_potentials, table_width=len(self.pot_r) + 1, engine=engine)

    def plot(self):
        """ """
        for pair in self.pairs:
            for n in range(self.n_iterations):
                potential_file = os.path.join(self.potentials_dir, 'step{0:d}.{1}'.format(
                    n, os.path.basename(pair.potential_file)))
                data = np.loadtxt(potential_file)
                plt.plot(data[:, 0], data[:, 1], label='n={0:d}'.format(n))
            plt.xlabel('r')
            plt.ylabel('V(r)')
            plt.legend()
            plt.savefig('figures/{0}.pdf'.format(pair.name))

def run_query_simulations(states, engine='hoomd'):
    """Run all query simulations for a single iteration. """
    # TODO: GPU count and proper "cluster management"
    pool = Pool(mp.cpu_count())
    print("Launching {0:d} threads...".format(mp.cpu_count()))
    if engine.lower() == 'hoomd':
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)
    pool.imap(worker, states)
    pool.close()
    pool.join()


def _hoomd_worker(state):
    """Worker for managing a single HOOMD-blue simulation. """
    log_file = os.path.join(state.state_dir, 'log.txt')
    err_file = os.path.join(state.state_dir, 'err.txt')
    with open(log_file, 'w') as log, open(err_file, 'w') as err:
        proc = Popen(['hoomd', 'run.py'], cwd=state.state_dir, stdout=log,
                     stderr=err, universal_newlines=True)
        print("    Launched HOOMD in {0}...".format(state.state_dir))
        proc.communicate()
        print("    Finished in {0}.".format(state.state_dir))
    _post_query(state)


def _post_query(state):
    state.reload_query_trajectory()

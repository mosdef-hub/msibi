import multiprocessing as mp
from multiprocessing.dummy import Pool
import os
from subprocess import Popen

import numpy as np

from msibi.utils.exceptions import UnsupportedEngine


class MSIBI(object):
    """
    """

    def __init__(self, rdf_cutoff, dr, pot_cutoff=None):
        self.states = None
        self.pairs = None
        self.rdf_cutoff = rdf_cutoff
        self.dr = dr
        self.rdf_r = np.arange(0.0, rdf_cutoff + 0.5 * dr, dr)

        # TODO: description of use for pot vs rdf cutoff
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff
        self.pot_r = np.arange(0.0, pot_cutoff + 0.5 * dr, dr)

    def optimize(self, states, pairs, engine='hoomd'):
        """
        """
        self.states = states
        self.pairs = pairs
        self.initialize(engine=engine)
        for n in range(10):
            run_query_simulations(self.states, engine=engine)

            for pair in self.pairs:
                for state in pair.states:
                    pair.compute_current_rdf(state, np.array([0.0, self.rdf_cutoff]), self.dr)
                pair.update_potential()
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
            potentials_dir = os.path.join(os.getcwd(), 'potentials')
        try:
            os.mkdir(potentials_dir)
        except OSError:
            # TODO: warning and maybe a "make backups" feature
            pass

        table_potentials = []
        for pair in self.pairs:
            potential_file = os.path.join(potentials_dir, 'pot.{0}.txt'.format(pair.name))
            table_potentials.append((pair.type1, pair.type2, potential_file))

            # This file is written for later viewing of how the potential evolves.
            pair.save_table_potential(potential_file, self.pot_r, self.dr, iteration=0, engine=engine)
            # This file is overwritten at each iteration and actually used for the
            # simulation.
            pair.save_table_potential(potential_file, self.pot_r, self.dr, engine=engine)

        for state in self.states:
            state.save_runscript(table_potentials, table_width=len(self.pot_r), engine=engine)


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

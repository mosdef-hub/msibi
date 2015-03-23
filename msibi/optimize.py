from subprocess import Popen, PIPE
import os

import numpy as np

R_RANGE = [0.0, 2.0]
DR = 0.01
R = np.arange(R_RANGE[0], R_RANGE[1] + 0.5 * DR, DR)

from msibi.pair import Pair
from msibi.state import State


def optimize(states, pairs):
    """
    """
    initialize(states, pairs, engine='hoomd')
    for n in range(10):
        run_queries(states)

        for pair in pairs:
            for state in pair.states:
                pair.compute_current_rdf(state)
            pair.update_potential()
        print("Finished iteration {0}".format(n))

def initialize(states, pairs, engine='hoomd', potentials_dir=None):
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
    for pair in pairs:
        potential_file = os.path.join(potentials_dir, 'pot.{0}.txt'.format(pair.name))
        table_potentials.append((pair.type1, pair.type2, potential_file))

        # This file is written for later viewing of how the potential evolves.
        pair.save_table_potential(potential_file, iteration=0)
        # This file is overwritten at each iteration and actually used for the
        # simulation.
        pair.save_table_potential(potential_file)

    for state in states:
        state.save_runscript(table_potentials, engine=engine)


def run_queries(states):
    for state in states:
        """
        proc = Popen('hoomd', cwd=state.state_dir, stdout=PIPE,
                     stderr=PIPE, universal_newlines=True, stdin=PIPE)
        out, err = proc.communicate('run.py')
        print out, err
        """
        os.chdir(state.state_dir)
        os.system('hoomd run.py > log.txt')
        os.chdir(os.pardir)


        state.reload_query_trajectory()


if __name__ == "__main__":
    # Load states
    state0 = State(k=5, T=1.0, traj_file='query.dcd', top_file='query.pdb')
    states = [state0, state1]

    # Creating pairs
    indices = [a.index for a in state0.traj.top._atoms]
    target = np.loadtxt('rdf.txt')
    pair0 = Pair('HC-CH', target)

    pairs = [pair0, pair3]

    # Add pairs to relevant states.
    pair0.add_state(state0, target_rdf0, alpha0, pair_indices0)

    msibi.optimize(states, pairs)
